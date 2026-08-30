from . import RLAgent
from common.registry import Registry
import numpy as np
import os
import atexit
import random
from collections import OrderedDict, deque
import gym

from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator
from transfer.observation import (
    build_divisors as build_observation_divisors,
    summarize as summarize_capacity,
    DEFAULT_CLIP as OBS_CAPACITY_CLIP,
)
import torch
from torch import nn
import torch.nn.functional as F
import torch_scatter
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops


@Registry.register_model('colight')
class CoLightAgent(RLAgent):
    #  TODO: test multiprocessing effect on agents or need deep copy here
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        """
        multi-agents in one model-> modify self.action_space, self.reward_generator, self.ob_generator here
        """
        #  general setting of world and model structure
        # TODO: different phases matching
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph
        self.world = world
        self.sub_agents = len(self.world.intersections)
        # TODO: support dynamic graph later
        self.colight_adjacency = str(Registry.mapping['model_mapping']['setting'].param.get(
            'colight_adjacency', 'road')).lower()
        if self.colight_adjacency not in ('road', 'contracted'):
            raise ValueError(f'Unknown colight_adjacency: {self.colight_adjacency}')
        # sparse_adj is indexed by graph node index, which is the order the
        # intersections appear in the converted roadnet JSON.  Everything else
        # this agent holds -- observations, rewards, phase_lengths, the action
        # vector the trainer applies -- is indexed by world.intersections.  On
        # the CityFlow grids the two orders coincide and the distinction never
        # surfaces; on Ingolstadt21 all 21 rows are out of place, so without
        # this remap every node aggregates the neighbours of an unrelated
        # intersection.  world order is authoritative, so the adjacency moves to
        # match it; where the orders already agree the remap is the identity.
        self.edge_idx = torch.tensor(
            self._world_ordered_adjacency().T, dtype=torch.long)  # source -> target

        #  model parameters
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['setting'].param

        #  get generator for CoLightAgent
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        # Deliberately left in world.intersections order; the adjacency is
        # brought to this order by _world_ordered_adjacency instead.
        self.ob_generator = observation_generators

        #  get reward generator for CoLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        self.reward_generator = rewarding_generators

        #  get queue generator for CoLightAgent
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"], 
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        self.queue = queues

        #  get delay generator for CoLightAgent
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"], 
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        self.delay = delays

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        self.phase_generator = phasing_generators

        # TODO: add irregular control of signals in the future
        self.phase_lengths = np.array([len(i.phases) for i in self.world.intersections])
        self.action_space = gym.spaces.Discrete(max(self.phase_lengths))
        min_ob_length = max([ob[1].ob_length for ob in self.ob_generator])
        # The lane block is padded to one common width and every other block is
        # appended after it.  Padding straight out to ob_length instead would
        # leave the phase block at a different offset for each intersection --
        # the one misalignment a shared network cannot undo, since it receives
        # no per-intersection identity input.
        self.lane_dim = int(min_ob_length)
        if self.phase:
            if self.one_hot:
                # max(phase_lengths), not intersections[0].phases: on a
                # heterogeneous network the first intersection is not the
                # widest, so a one-hot sized from it silently truncates the
                # phase of every intersection that has more.  On the CityFlow
                # grids the two are equal and this changes nothing.
                self.phase_dim = int(self.phase_lengths.max())
            else:
                self.phase_dim = 1
        else:
            self.phase_dim = 0
        self.ob_length = self.lane_dim + self.phase_dim

        # Capacity normalisation, as ported for HyperLight in
        # transfer/observation.py: divide each lane's count by what that lane
        # can physically hold rather than by one global constant.  vehicle_max
        # is 1 in colight.yml, i.e. the observation is a raw vehicle count, so
        # on a network whose lanes differ in length the same input value means
        # different things at different intersections -- again something a
        # shared network with no identity input cannot separate.
        self.obs_norm_mode = str(Registry.mapping['model_mapping']['setting'].param.get(
            'colight_obs_norm', 'fixed')).lower()
        if self.obs_norm_mode not in ('fixed', 'capacity'):
            raise ValueError(f'Unknown colight_obs_norm: {self.obs_norm_mode}')
        self.obs_divisors = None
        if self.obs_norm_mode == 'capacity':
            fallback = float(Registry.mapping['model_mapping']['setting'].param['vehicle_max'])
            divisors, resolved_total, missing_total = [], 0, 0
            for _, ob_gen in self.ob_generator:
                lane_ids = [lane for road_lanes in ob_gen.lanes for lane in road_lanes]
                node_divisors, resolved, missing = build_observation_divisors(
                    self.world, lane_ids, self.lane_dim, 1, fallback=fallback)
                divisors.append(node_divisors)
                resolved_total += resolved
                missing_total += missing
            self.obs_divisors = np.stack(divisors).astype(np.float32)
            note = summarize_capacity(divisors)
            if missing_total:
                note += (f' [{missing_total}/{resolved_total + missing_total} lanes had '
                         f'no length; fell back to vehicle_max]')
            print(f'[colight] {note}', flush=True)

        self.get_attention = Registry.mapping['logger_mapping']['setting'].param['attention']
        # train parameters
        self.rank = rank
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']

        # Diagnostic: which phase each intersection actually selects while
        # acting greedily.  A shared head that has collapsed onto one phase per
        # intersection looks the same in the travel-time column as a policy that
        # is merely mediocre, and the two call for different fixes.
        self.log_phase_hist = bool(Registry.mapping['model_mapping']['setting'].param.get(
            'colight_phase_hist', False))
        self.phase_hist = np.zeros((self.sub_agents, int(self.action_space.n)), dtype=np.int64)
        if self.log_phase_hist:
            atexit.register(self.report_phase_hist)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)

    def _world_ordered_adjacency(self):
        """The chosen adjacency, reindexed from graph order to world order.

        ``colight_adjacency`` picks which one:

        road        an edge where a single road joins two signals.  The default,
                    and what every result before this option was produced with.
        contracted  an edge where a path joins two signals through junctions
                    that carry no signal.  On the CityFlow grids this is exactly
                    the same set of edges, because there the signals really are
                    joined by single roads.  On Ingolstadt21 it is the
                    difference between 2 edges and 143: under `road`, 19 of 21
                    nodes have degree zero and attend only to themselves, so
                    CoLight is 21 independent agents wearing a graph.
        """
        mode = getattr(self, 'colight_adjacency', 'road')
        key = 'sparse_adj' if mode == 'road' else 'sparse_adj_reachable'
        if key not in self.graph:
            raise ValueError(
                f'graph has no {key!r}; colight_adjacency={mode} is unavailable '
                'for this world')
        adjacency = np.asarray(self.graph[key], dtype=np.int64).reshape(-1, 2)
        idx2id = self.graph['node_idx2id']
        world_pos = {}
        for pos, inter in enumerate(self.world.intersections):
            node_id = inter.id[3:] if inter.id.startswith('GS_') else inter.id
            world_pos[node_id] = pos
        missing = [idx2id[g] for g in range(len(idx2id)) if idx2id[g] not in world_pos]
        if missing:
            raise ValueError(
                'graph nodes absent from world.intersections: ' + ', '.join(map(str, missing)))
        remap = np.array([world_pos[idx2id[g]] for g in range(len(idx2id))], dtype=np.int64)
        return remap[adjacency]

    def reset(self):
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        self.ob_generator = observation_generators

        #  get reward generator for CoLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        self.reward_generator = rewarding_generators

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        self.phase_generator = phasing_generators

        # queue metric
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"], 
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        self.queue = queues

        # delay metric
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"], 
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        self.delay = delays

    def get_ob(self):
        x_obs = []  # sub_agents * lane_nums,
        for i in range(len(self.ob_generator)):
            ob = self.ob_generator[i][1].generate()/ self.vehicle_max
            ob = np.pad(ob, (0, self.lane_dim - ob.shape[-1] ))
            if self.obs_divisors is not None:
                ob = np.clip(ob / self.obs_divisors[i], 0.0, OBS_CAPACITY_CLIP)
            x_obs.append(ob)

        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def _with_phase(self, ob, phase):
        """Append the current-phase block to a [agents, lane_dim] observation.

        ``get_action`` was handed ``phase`` and threw it away (the old
        ``# TODO: not phase not used``), so the network never saw which phase it
        was currently running.  On the CityFlow grids that is survivable: every
        intersection shares one 4-phase convention, so phase index k means the
        same movement everywhere and the lane counts alone carry most of the
        state.  On a network with unequal phase counts index k means a different
        movement at each intersection, and without this block the shared network
        is asked to choose a phase without knowing which one is running.
        """
        if self.phase_dim == 0:
            return ob
        ob = np.asarray(ob, dtype=np.float32)
        phase = np.asarray(phase).reshape(-1).astype(np.int64)
        if self.one_hot:
            block = np.zeros((ob.shape[0], self.phase_dim), dtype=np.float32)
            # Clip rather than trust the index: a phase id at or beyond that
            # intersection's phase count would otherwise write out of bounds.
            idx = np.clip(phase, 0, self.phase_dim - 1)
            block[np.arange(ob.shape[0]), idx] = 1.0
        else:
            block = (phase / np.maximum(self.phase_lengths - 1, 1)).astype(
                np.float32).reshape(-1, 1)
        return np.concatenate([ob, block], axis=-1)

    def get_reward(self):
        # TODO: test output
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):
            rewards.append(self.reward_generator[i][1].generate())
        rewards = np.squeeze(np.array(rewards, dtype=np.float32)) * 12
        return rewards

    def get_phase(self):
        # TODO: test phase output onehot/int
        phase = []  # sub_agents
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))
        phase = (np.concatenate(phase)).astype(np.int8)
        # phase = np.concatenate(phase, dtype=np.int8)
        return phase

    def get_queue(self):
        """
        get delay of intersection
        return: value(one intersection) or [intersections,](multiple intersections)
        """
        queue = []
        for item in self.queue:
            item = item[1].generate()
            item = np.pad(item, (0, self.lane_dim - item.shape[-1]))
            queue.append(item)
            
        tmp_queue = np.squeeze(np.array(queue, dtype=np.float32))
        queue = np.sum(tmp_queue, axis=1 if len(tmp_queue.shape)==2 else 0)
        return queue

    def get_delay(self):
        delay = []
        for i in range(len(self.delay)):
            delay.append((self.delay[i][1].generate()))
        delay = np.squeeze(np.array(delay, dtype=np.float32))
        return delay # [intersections,]

    def get_action(self, ob, phase, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length] -> [batch, agents, ob_length]
        :param phase: [agents] -> [batch, agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [batch, agents] -> action taken by environment
        """
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        observation = torch.tensor(self._with_phase(ob, phase), dtype=torch.float32)
        edge = self.edge_idx
        dp = Data(x=observation, edge_index=edge)

        if self.get_attention:
            # TODO: collect attention matrix later
            actions = self.model(x=dp.x, edge_index=dp.edge_index, train=False)
            att = None
            actions = actions.clone().detach().numpy()
            # action = np.argmax(actions, axis=1)
            action_list = []
            for action_vec, phase_length in zip(actions, self.phase_lengths):
                action_list.append(np.argmax(action_vec[0:phase_length]))
            # action = np.clip(action, 0, self.phase_lengths - 1)
            action = self._record_phase_hist(np.array(action_list), test)
            # action = np.clip(action, 0, self.phase_lengths - 1)
            return action, att  # [batch, agents], [batch, agents, nv, neighbor]
        else:
            actions = self.model(x=dp.x, edge_index=dp.edge_index, train=False)
            actions = actions.clone().detach().numpy()
            
            action_list = []
            for action_vec, phase_length in zip(actions, self.phase_lengths):
                action_list.append(np.argmax(action_vec[0:phase_length]))
            # action = np.clip(action, 0, self.phase_lengths - 1)
            action = self._record_phase_hist(np.array(action_list), test)
            
            return action  # [batch, agents] TODO: check here

    def _record_phase_hist(self, action, test):
        """Count greedy phase choices per intersection; returns action unchanged.

        Only test-time actions are counted: during training epsilon starts at
        0.8, so a training histogram mostly measures the exploration schedule.
        """
        if self.log_phase_hist and test:
            np.add.at(self.phase_hist, (np.arange(action.shape[0]), action), 1)
        return action

    def report_phase_hist(self):
        """Print the per-intersection phase distribution collected so far."""
        totals = self.phase_hist.sum(axis=1)
        if not totals.any():
            return
        print('[colight] greedy phase distribution per intersection '
              '(share of test-time decisions):', flush=True)
        collapsed = 0
        for idx, inter in enumerate(self.world.intersections):
            total = totals[idx]
            if total == 0:
                continue
            n_phases = int(self.phase_lengths[idx])
            shares = self.phase_hist[idx, :n_phases] / float(total)
            if shares.max() > 0.95:
                collapsed += 1
            body = ' '.join(f'{share:.2f}' for share in shares)
            print(f'[colight]   {inter.id:<20s} n={n_phases}  {body}', flush=True)
        print(f'[colight] {collapsed}/{len(totals)} intersections spend >95% of '
              f'their decisions in a single phase', flush=True)

    def sample(self):
        action = np.random.randint(0, self.action_space.n, self.sub_agents)
        action = np.clip(action, 0, self.phase_lengths - 1)
        return action

    def _build_model(self):
        model = ColightNet(self.ob_length, self.action_space.n, self.phase_lengths, **self.model_dict)
        return model

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

    def _batchwise(self, samples):
        # load onto tensor

        batch_list = []
        batch_list_p = []
        actions = []
        rewards = []
        for item in samples:
            dp = item[1]
            # dp[1] / dp[5] are the phases stored by remember(); they were kept
            # in the buffer but never used, so the replayed input has to be
            # rebuilt the same way get_action builds the acting input.
            state = torch.tensor(self._with_phase(dp[0], dp[1]), dtype=torch.float32)
            batch_list.append(Data(x=state, edge_index=self.edge_idx))

            state_p = torch.tensor(self._with_phase(dp[4], dp[5]), dtype=torch.float32)
            batch_list_p.append(Data(x=state_p, edge_index=self.edge_idx))
            rewards.append(dp[3])
            actions.append(dp[2])
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        if self.sub_agents > 1:
            rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
            actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here
        # rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        # actions = torch.tensor(np.array(actions), dtype=torch.long)
        # actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here

        return batch_t, batch_tp, rewards, actions

    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)

        out = self.target_model(x=b_tp.x, edge_index=b_tp.edge_index, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model(x=b_t.x, edge_index=b_t.edge_index, train=False)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model(x=b_t.x, edge_index=b_t.edge_index, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().detach().numpy()

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                'model', f'{e}_{self.rank}.pt')
        self.model.load_state_dict(torch.load(model_name))
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)


class ColightNet(nn.Module):
    def __init__(self, input_dim, output_dim, phase_lengths, **kwargs):
        super(ColightNet, self).__init__()
        self.model_dict = kwargs
        self.batch_size = self.model_dict['batch_size']
        self.action_space = gym.spaces.Discrete(output_dim)
        self.features = input_dim
        self.module_list = nn.ModuleList()
        self.embedding_MLP = Embedding_MLP(self.features, layers=self.model_dict.get('NODE_EMB_DIM'))
        for i in range(self.model_dict.get('N_LAYERS')):
            block = MultiHeadAttModel(d=self.model_dict.get('INPUT_DIM')[i],
                                      dv=self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                      d_out=self.model_dict.get('OUTPUT_DIM')[i],
                                      nv=self.model_dict.get('NUM_HEADS')[i],
                                      suffix=i)
            self.module_list.append(block)
        output_dict = OrderedDict()

        if len(self.model_dict['OUTPUT_LAYERS']) != 0:
            # TODO: dubug this branch
            for l_idx, l_size in enumerate(self.model_dict['OUTPUT_LAYERS']):
                name = f'output_{l_idx}'
                if l_idx == 0:
                    h = nn.Linear(block.d_out, l_size)
                else:
                    h = nn.Linear(self.model_dict.get('OUTPUT_LAYERS')[l_idx - 1], l_size)
                output_dict.update({name: h})
                name = f'relu_{l_idx}'
                output_dict.update({name: nn.ReLU})
            out = nn.Linear(self.model_dict['OUTPUT_LAYERS'][-1], self.action_space.n)
        else:
            out = nn.Linear(block.d_out, self.action_space.n)
        name = f'output'
        output_dict.update({name: out})
        
        # make mask
        unpadded_phase_mask = [torch.ones(length, dtype=torch.bool) for length in phase_lengths]
        phase_mask = torch.nn.utils.rnn.pad_sequence(unpadded_phase_mask, batch_first=True)
        mask_layer = MaskedOutput(mask=phase_mask, batch_size=self.batch_size, action_space=self.action_space)
        output_dict.update({'out_mask': mask_layer})

        self.output_layer = nn.Sequential(output_dict)

    def forward(self, x, edge_index, train=True):
        h = self.embedding_MLP.forward(x, train)
        # TODO: implement att

        if train:
            for mdl in self.module_list:
                h = mdl.forward(h, edge_index, train)
            h = self.output_layer(h)
        else:
            with torch.no_grad():
                for mdl in self.module_list:
                    h = mdl.forward(h, edge_index, train)
                h = self.output_layer(h)
        return h

#: Value written into the Q slot of a phase an intersection does not have.
#: It only has to sit below every reachable Q-value: rewards are a negative
#: lane count (scaled by 12), so with gamma=0.95 a real Q is on the order of
#: -1e2..-1e3.  A finite sentinel rather than -inf because ``train`` feeds this
#: same output in as its MSE target, and -inf would make those entries nan.
INVALID_ACTION_Q = -1e6


class MaskedOutput(nn.Module):
    """Force the Q of phases an intersection does not have below every real one.

    This used to *multiply* by the 0/1 mask, which drove invalid phases to
    exactly 0.  Acting was unaffected -- ``get_action`` slices
    ``action_vec[0:phase_length]`` before its argmax -- but ``train`` does

        target = rewards + gamma * torch.max(out, dim=1)[0]

    over the full padded row.  Rewards here are negative, so every real Q is
    negative and that 0 always won the max: the bootstrap term collapsed to 0 at
    every intersection with fewer than ``max(phase_lengths)`` phases, which is
    an effective gamma of 0 -- a purely myopic agent.  On the homogeneous
    CityFlow grids the mask is all ones and nothing happens; on Ingolstadt21 it
    hit 15 of 21 intersections.
    """

    def __init__(self, mask, batch_size, action_space):
        super(MaskedOutput, self).__init__()
        self.batch_size = batch_size
        self.mask = mask
        self.action_space = action_space

    def forward(self, x):
        masked_output = x.reshape(-1, self.mask.shape[0], self.action_space.n)
        masked_output = masked_output.masked_fill(~self.mask, INVALID_ACTION_Q)
        masked_output = masked_output.reshape(-1, self.mask.shape[-1])
        return masked_output

class Embedding_MLP(nn.Module):
    def __init__(self, in_size, layers):
        super(Embedding_MLP, self).__init__()
        constructor_dict = OrderedDict()
        for l_idx, l_size in enumerate(layers):
            name = f"node_embedding_{l_idx}"
            if l_idx == 0:
                h = nn.Linear(in_size, l_size)
                constructor_dict.update({name: h})
            else:
                h = nn.Linear(layers[l_idx - 1], l_size)
                constructor_dict.update({name: h})
            name = f"n_relu_{l_idx}"
            constructor_dict.update({name: nn.ReLU()})
        self.embedding_node = nn.Sequential(constructor_dict)

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class MultiHeadAttModel(MessagePassing):
    """
    inputs:
        In_agent [bacth,agents,128]
        In_neighbor [agents, neighbor_num]
        l: number of neighborhoods (in my code, l=num_neighbor+1,because l include itself)
        d: dimension of agents's embedding
        dv: dimension of each head
        dout: dimension of output
        nv: number of head (multi-head attention)
    output:
        -hidden state: [batch,agents,32]
        -attention: [batch,agents,neighbor]
    """
    def __init__(self, d, dv, d_out, nv, suffix):
        super(MultiHeadAttModel, self).__init__(aggr='add')
        self.d = d
        self.dv = dv
        self.d_out = d_out
        self.nv = nv
        self.suffix = suffix
        # target is center
        self.W_target = nn.Linear(d, dv * nv)
        self.W_source = nn.Linear(d, dv * nv)
        self.hidden_embedding = nn.Linear(d, dv * nv)
        self.out = nn.Linear(dv, d_out)
        self.att_list = []
        self.att = None

    # Some PyG versions generate a subclass-level propagate() signature that
    # does not expose kwargs (e.g., x=...), while MessagePassing.propagate does.
    # Route through the base implementation to keep old call-sites compatible.
    def propagate(self, edge_index, size=None, x=None):
        return MessagePassing.propagate(self, edge_index=edge_index, size=size, x=x)

    def _forward(self, x, edge_index):
        # TODO: test batch is shared or not

        # x has shape [N, d], edge_index has shape [E, 2]
        edge_index, _ = add_self_loops(edge_index=edge_index)
        aggregated = self.propagate(x=x, edge_index=edge_index)  # [16, 16]
        out = self.out(aggregated)
        out = F.relu(out)  # [ 16, 128]
        #self.att = torch.tensor(self.att_list)
        return out

    def forward(self, x, edge_index, train=True):
        if train:
            return self._forward(x, edge_index)
        else:
            with torch.no_grad():
                return self._forward(x, edge_index)

    def message(self, x_i, x_j, edge_index):
        h_target = F.relu(self.W_target(x_i))
        h_target = h_target.view(h_target.shape[:-1][0], self.nv, self.dv)
        agent_repr = h_target.permute(1, 0, 2)

        h_source = F.relu(self.W_source(x_j))
        h_source = h_source.view(h_source.shape[:-1][0], self.nv, self.dv)

        neighbor_repr = h_source.permute(1, 0, 2)  # [nv, E, dv]
        index = edge_index[1]  # which is target

        e_i = torch.mul(agent_repr, neighbor_repr).sum(-1)  # [5, 64]
        max_node = torch_scatter.scatter_max(e_i, index=index)[0]  # [5, 16]
        max_i = max_node.index_select(1, index=index)  # [5, 64]
        ec_i = torch.add(e_i, -max_i)
        ecexp_i = torch.exp(ec_i)
        norm_node = torch_scatter.scatter_sum(ecexp_i, index=index)  # [5, 16]
        normst_node = torch.add(norm_node, 1e-12)  # [5, 16]
        normst_i = normst_node.index_select(1, index)  # [5, 64]

        alpha_i = ecexp_i / normst_i  # [5, 64]
        alpha_i_expand = alpha_i.repeat(self.dv, 1, 1)
        alpha_i_expand = alpha_i_expand.permute((1, 2, 0))  # [5, 64, 16]

        hidden_neighbor = F.relu(self.hidden_embedding(x_j))
        hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv)
        hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2)  # [5, 64, 16]
        out = torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0)

        # TODO: attention ouput in the future
        # self.att_list.append(alpha_i)  # [64, 16]
        return out

    def get_att(self):
        if self.att is None:
            print('invalid att')
        return self.att
