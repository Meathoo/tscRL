import task
import trainer
import agent
import dataset
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import *
import time
from datetime import datetime
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# parseargs
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--thread_num', type=int, default=4, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="-1", help='gpu to be used')  # choose gpu card
parser.add_argument('--prefix', type=str, default='test', help="the number of prefix in this running process")
parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")
parser.add_argument('--debug', nargs='?', const=True, default=False, type=str2bool)
parser.add_argument('--no-debug', action='store_false', dest='debug')
parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo','traci'], help="interface type") # libsumo(fast) or traci(slow)
parser.add_argument('--delay_type', type=str, default="apx", choices=['apx','real'], help="method of calculating delay") # apx(approximate) or real

parser.add_argument('-t', '--task', type=str, default="tsc", help="task type to run")
parser.add_argument('-a', '--agent', type=str, default="dqn", help="agent type of agents in RL environment")
parser.add_argument('-w', '--world', type=str, default="cityflow", choices=['cityflow','sumo'], help="simulator type")
parser.add_argument('-n', '--network', type=str, default="cityflow1x1", help="network name")
parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')
parser.add_argument(
    '--agent_embedding_mode',
    type=str,
    default=None,
    choices=[
        'one_hot',
        'learned',
        'topology',
        'learned_topology',
        'one_hot_topology',
        'structural',
        'constant',
        'frozen',
    ],
    help='override model.agent_embedding_mode; "structural" drops the '
         'per-intersection index table so the meta vector transfers across '
         'road networks (see transfer/TRANSFER.md). "frozen" is learned\'s '
         'exact shape and cardinality with the codes never trained, which '
         'separates "the codes are per-intersection" from "the codes have to '
         'be learned from the RL signal" as reasons learned underperforms',
)
parser.add_argument(
    '--structural_features',
    type=str,
    default=None,
    help='override model.structural_features (agent_embedding_mode=structural '
         'only): comma-separated subset of the structural contract, e.g. '
         '"in_lane_count,out_lane_count,in_degree,out_degree". Omit for the '
         'full 12-feature contract. The subset changes spec_id(), so a subset '
         'run cannot silently load a full-contract checkpoint.',
)
parser.add_argument(
    '--structural_shrink',
    type=float,
    default=None,
    help='override model.structural_shrink (agent_embedding_mode=structural '
         'only): pull every structural feature toward this network\'s own mean, '
         'x -> mean + shrink*(x - mean). 1.0 (the default) is exactly inert; '
         '0.0 makes every intersection identical, i.e. the constant-meta arm '
         'with the structural code path. It varies the MAGNITUDE of structural '
         'variation while holding simulator, city and flow fixed, which no pair '
         'of networks in the tree does. Shrunk features depend on the loaded '
         'roadnet, so spec_id() flags them and a shrunk checkpoint cannot be '
         'used as a transfer source.',
)
parser.add_argument('--movement_encoder_enabled', type=str2bool, nargs='?', const=True,
                    default=None,
                    help='override model.movement_encoder_enabled: encode the per-lane '
                         'observation as masked movement tokens, giving the actor a width '
                         'that does not depend on the lane count. Needed to transfer '
                         'between networks whose intersections differ in size (blocker B4 '
                         'in transfer/TRANSFER.md). Never exercised before 2026-08-28.')
parser.add_argument('--movement_phase_head', type=str2bool, nargs='?', const=True,
                    default=None,
                    help='override model.movement_phase_head: score each phase from the '
                         'movement tokens it gives green to, instead of emitting one logit '
                         'per action index. Removes the phase count from every parameter '
                         'shape, which is what lets a checkpoint cross networks that signal '
                         'differently (blocker B4, output half). Requires '
                         '--movement_encoder_enabled.')
parser.add_argument('--sumo_teleport_time', type=int, default=None,
                    help='pass --time-to-teleport to SUMO. Omitted by default, so '
                         'SUMO applies its own default of 300s -- which is what every '
                         'result in this repo was produced under. RESCO-derived work '
                         'disables teleporting (-1); Unicorn uses -1 for RESCO maps, '
                         '300 for MA2C/SG and 600 for GESA. Teleporting removes stuck '
                         'vehicles, so it moves the mean travel time and the completed '
                         'vehicle count together. SUMO world only.')
parser.add_argument('--sumo_yellow_length', type=float, default=None,
                    help='override the yellow phase length. By default it is read from '
                         'the .net.xml (the shortest phase in the signal\'s own '
                         'programme, mostly 3s on Ingolstadt21) and the yellow_length '
                         'keys in configs/ have never had any effect. RESCO and Unicorn '
                         'use 5. SUMO world only.')
parser.add_argument('--colight_adjacency', type=str, default=None,
                    choices=['road', 'contracted'],
                    help='override model.colight_adjacency: how two signals count as '
                         'neighbours. road = one road joins them (default, and what '
                         'existing results used); contracted = a path joins them through '
                         'junctions carrying no signal. Identical on the CityFlow grids.')
parser.add_argument('--colight_obs_norm', type=str, default=None,
                    choices=['fixed', 'capacity'],
                    help='override model.colight_obs_norm: CoLight divides its lane '
                         'counts by vehicle_max (1 in colight.yml, i.e. no normalisation '
                         'at all). "capacity" divides by each lane own storage instead, '
                         'so a reading means the same on a short and a long approach. '
                         'Identical in effect on networks with uniform lane lengths.')
parser.add_argument('--colight_phase', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.phase for CoLight: feed the current phase into '
                         'the shared network. It was accepted by get_action and dropped, '
                         'so it has never reached the model. Matters only where phase '
                         'index k means different movements at different intersections.')
parser.add_argument('--load_model', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.load_model; with --train_model False this is what '
                         'makes task.py load the saved best checkpoint before testing. '
                         'Without it an evaluation-only run silently scores a freshly '
                         'initialised network. Agents that take --transfer_checkpoint '
                         'load their weights at construction and do not need this.')
parser.add_argument('--colight_phase_hist', type=str2bool, nargs='?', const=True, default=None,
                    help='print, at exit, how each intersection distributes its greedy '
                         'test-time decisions over its phases. Separates a policy that '
                         'has collapsed onto one phase per intersection from one that is '
                         'merely mediocre; the travel-time column cannot tell them apart.')
parser.add_argument('--hypernet_type', type=str, default=None, choices=['mlp', 'linear'],
                    help='override model hypernetwork type')
parser.add_argument('--hyper_actor_arch', type=str, default=None, choices=['mlp', 'iru'],
                    help='override model.hyper_actor_arch')
parser.add_argument('--hyper_actor_hidden1', type=int, default=None,
                    help='override model.actor_hidden1 for HyperLight MLP controls')
parser.add_argument('--hyper_actor_hidden2', type=int, default=None,
                    help='override model.actor_hidden2 for HyperLight MLP controls')
parser.add_argument('--hyper_adapter_mode', type=str, default=None,
                    choices=['generated', 'film', 'none'],
                    help='override actor adaptation: generated weights, FiLM, or shared actor')
parser.add_argument('--hyper_critic_adapter_mode', type=str, default=None,
                    choices=['generated', 'film'],
                    help='override critic adaptation: generated weights or shared critic with FiLM')
parser.add_argument('--hyper_film_scale', type=float, default=None,
                    help='override model.hyper_film_scale')
parser.add_argument('--reward_mode', type=str, default=None,
                    choices=['queue', 'pressure_abs', 'queue_pressure', 'pressure', 'waiting', 'mean_waiting', 'mplight'],
                    help='override model.reward_mode')
parser.add_argument('--pressure_balance_coef', type=float, default=None,
                    help='override model.pressure_balance_coef')
parser.add_argument('--native_use_agent_id', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.native_use_agent_id')
parser.add_argument('--native_agent_id_mode', type=str, default=None,
                    choices=['one_hot', 'learned'],
                    help='override model.native_agent_id_mode')
parser.add_argument('--native_actor_arch', type=str, default=None, choices=['mlp', 'iru'],
                    help='override model.native_actor_arch')
parser.add_argument('--native_value_arch', type=str, default=None, choices=['mlp', 'iru'],
                    help='override model.native_value_arch')
parser.add_argument('--iru_steps', type=int, default=None,
                    help='override both model.iru_actor_steps and model.iru_value_steps')
parser.add_argument('--iru_actor_steps', type=int, default=None,
                    help='override model.iru_actor_steps')
parser.add_argument('--iru_value_steps', type=int, default=None,
                    help='override model.iru_value_steps')
parser.add_argument('--iru_hidden_dim', type=int, default=None,
                    help='override model.iru_hidden_dim')
parser.add_argument('--iru_num_blocks', type=int, default=None,
                    help='override model.iru_num_blocks')
parser.add_argument('--profile_performance', type=str2bool, nargs='?', const=True, default=None,
                    help='record parameter count, latency, update time, and peak GPU memory')
parser.add_argument('--episodes', type=int, default=None,
                    help='override trainer.episodes (the final episode boundary)')
parser.add_argument('--resume_episode', type=int, default=None,
                    help='load model/<episode>_<rank>.pt before training and continue from this episode')
parser.add_argument('--config_snapshot', type=str, default=None,
                    help='rebuild model/trainer/logger/world settings from a saved hyperparameters.json')
parser.add_argument('--hyper_residual', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.hyper_residual')
parser.add_argument('--hyper_residual_mode', type=str, default=None,
                    choices=['full', 'lora', 'low_rank', 'low-rank', 'head', 'head_only', 'head-only',
                             'last_layer', 'last-layer'],
                    help='override model.hyper_residual_mode')
parser.add_argument('--hyper_residual_scale', type=float, default=None,
                    help='override model.hyper_residual_scale')
parser.add_argument('--hyper_residual_actor_scale', type=float, default=None,
                    help='override model.hyper_residual_actor_scale')
parser.add_argument('--hyper_residual_value_scale', type=float, default=None,
                    help='override model.hyper_residual_value_scale')
parser.add_argument('--hyper_lora_rank', type=int, default=None,
                    help='override model.hyper_lora_rank')
parser.add_argument('--hyper_lora_actor_rank', type=int, default=None,
                    help='override model.hyper_lora_actor_rank')
parser.add_argument('--hyper_lora_value_rank', type=int, default=None,
                    help='override model.hyper_lora_value_rank')
parser.add_argument('--hyper_lora_bias', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.hyper_lora_bias')
parser.add_argument('--hyper_head_mode', type=str, default=None,
                    choices=['flat', 'layerwise', 'chunked'],
                    help='override model.hyper_head_mode')
parser.add_argument('--hyper_chunk_size', type=int, default=None,
                    help='override model.hyper_chunk_size (chunked head only)')
parser.add_argument('--hyper_chunk_embed_dim', type=int, default=None,
                    help='override model.hyper_chunk_embed_dim (chunked head only)')
parser.add_argument('--hyper_actor_chunk_size', type=int, default=None,
                    help='override model.hyper_actor_chunk_size (defaults to hyper_chunk_size)')
parser.add_argument('--hyper_critic_chunk_size', type=int, default=None,
                    help='override model.hyper_critic_chunk_size (defaults to hyper_chunk_size)')
parser.add_argument('--hyper_chunk_generator_hidden', type=int, default=None,
                    help='override model.hyper_chunk_generator_hidden; 0 keeps the '
                         'single-Linear (purely additive) chunk generator')
parser.add_argument('--hyper_chunk_rf_mode', type=str, default=None,
                    choices=['shared', 'per_chunk'],
                    help='override model.hyper_chunk_rf_mode: how hyper_rf_init lays '
                         'the target-layer init into a chunked head. shared puts one '
                         'block in the generator bias that every chunk reads, so the '
                         'generated matrix starts rank-deficient; per_chunk slices a '
                         'full-size init across the chunks via the chunk codes at no '
                         'extra parameter cost (needs hyper_chunk_embed_dim >= n_chunks)')
parser.add_argument('--mixer_mode', type=str, default=None,
                    choices=['none', 'uniform', 'regime'],
                    help='override model.mixer_mode: replace the per-agent PPO objective '
                         'with a joint one decomposed by a monotonic mixer '
                         '(V_tot = sum_i w_i V_i + b, A_i = w_i A_tot). Requires '
                         'centralized_critic False. `uniform` is the fixed-w control; '
                         'mixer runs are not comparable with non-mixer numbers')
parser.add_argument('--mixer_regimes', type=int, default=None,
                    help='override model.mixer_regimes (K in the quantized regime code)')
parser.add_argument('--mixer_quantize', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.mixer_quantize; False keeps the code continuous, '
                         'which is the arm that tests whether quantization is what '
                         'separates this from the harmful conditioning measured in (h)')
parser.add_argument('--mixer_vq_coef', type=float, default=None,
                    help='override model.mixer_vq_coef')
parser.add_argument('--hyper_prototypes', type=int, default=None,
                    help='override model.hyper_prototypes: K in the prototype-factorized '
                         'head. theta_i becomes a convex mixture of K generated parameter '
                         'sets, with the mixing weights read off the same meta vector. '
                         '0 keeps the head unchanged; 1 is constmeta by construction, and '
                         'the learned embedding mode is the K=N end of the same axis')
parser.add_argument('--hyper_prototype_gate_hidden', type=int, default=None,
                    help='override model.hyper_prototype_gate_hidden; 0 uses a single Linear')
parser.add_argument('--hyper_prototype_temperature', type=float, default=None,
                    help='override model.hyper_prototype_temperature (gate softmax '
                         'temperature at the start of training)')
parser.add_argument('--hyper_prototype_temperature_final', type=float, default=None,
                    help='override model.hyper_prototype_temperature_final; the gate '
                         'sharpens linearly to this across the planned update count')
parser.add_argument('--hyper_prototype_gate_frozen', type=str2bool, nargs='?', const=True,
                    default=None,
                    help='override model.hyper_prototype_gate_frozen; freezes the gate at '
                         'its random init, giving a fixed arbitrary partition. Separates '
                         '"K sets of weights" from "alike intersections share a policy"')
parser.add_argument('--hyper_hidden', type=str, default=None,
                    help='override model.hyper_hidden, e.g. "256" or "128,64"')
parser.add_argument('--value_hyper_hidden', type=str, default=None,
                    help='override model.value_hyper_hidden; defaults to --hyper_hidden')
parser.add_argument('--agent_embedding_dim', type=int, default=None,
                    help='override model.agent_embedding_dim (agent_embedding_mode=learned only)')
parser.add_argument('--hyper_rf_init', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.hyper_rf_init (fan-in calibrated generator init)')
parser.add_argument('--save_rate', type=int, default=None,
                    help='override logger.save_rate (checkpoint every N episodes)')
parser.add_argument('--early_stop_patience', type=int, default=None,
                    help='override trainer.early_stop_patience; 0 disables early stopping. '
                         'ppo.yml sets 8, so the PPO-family baselines stop far short of the '
                         'episode budget the other methods run')
parser.add_argument('--action_interval', type=int, default=None,
                    help='override trainer.action_interval: seconds a chosen phase is held. '
                         'This is the green/phase duration, and it is what other studies vary '
                         'when they say "phase duration" -- RESCO and Unicorn use 15, this '
                         'repo defaults to 10. Changing it changes the decisions per episode '
                         '(steps / action_interval), so set --ppo_rollout_steps to match or '
                         'the update cadence moves with it.')
parser.add_argument('--ppo_rollout_steps', type=int, default=None,
                    help='override model.ppo_rollout_steps: transitions collected before a PPO '
                         'update. The default 360 equals 3600/10, i.e. exactly one update per '
                         'episode at the default action_interval.')
parser.add_argument('--gamma', type=float, default=None,
                    help='override model.gamma. base.yml uses 0.95 but hyperlight_ppo.yml sets '
                         '0.99, so the HyperLight family runs at 0.99 while RESCO-derived work '
                         'reports 0.95.')
parser.add_argument('--lr_anneal', type=str, default=None, choices=['none', 'linear'],
                    help='override model.lr_anneal; linear decays the learning rate to '
                         'lr_final_frac across the planned update count')
parser.add_argument('--entropy_anneal', type=str, default=None, choices=['none', 'linear'],
                    help='override model.entropy_anneal')
parser.add_argument('--obs_norm_mode', type=str, default=None, choices=['fixed', 'capacity'],
                    help='override model.obs_norm_mode; "capacity" divides each per-lane '
                         'count by that lane storage (length/headway) instead of by the '
                         'global vehicle_max, making readings comparable across lane lengths')
parser.add_argument('--obs_capacity_headway', type=float, default=None,
                    help='override model.obs_capacity_headway (metres per stopped vehicle)')
parser.add_argument('--dynamic_condition_enabled', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.dynamic_condition_enabled; conditions the '
                         'hypernetwork on a slow EMA of each intersection traffic state')
parser.add_argument('--dynamic_ema_halflife', type=float, default=None,
                    help='override model.dynamic_ema_halflife (in decision steps)')
parser.add_argument('--dynamic_hidden_dim', type=int, default=None,
                    help='override model.dynamic_hidden_dim; 0 uses a single Linear')
parser.add_argument('--dynamic_scale', type=float, default=None,
                    help='override model.dynamic_scale')
parser.add_argument('--train_model', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.train_model; False evaluates the agent as constructed '
                         '(with --transfer_checkpoint this is a true zero-shot evaluation)')
parser.add_argument('--transfer_checkpoint', type=str, default=None,
                    help='path to a checkpoint trained on ANOTHER road network; '
                         'shape-compatible weights are reused, per-index embeddings '
                         'and optimizer state are not (see transfer/TRANSFER.md)')
parser.add_argument('--transfer_strict', type=str2bool, nargs='?', const=True, default=None,
                    help='fail instead of warn when a transfer checkpoint leaves any '
                         'parameter uninitialised')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

logging_level = logging.INFO
if args.debug:
    logging_level = logging.DEBUG


class Runner:
    def __init__(self, pArgs):
        """
        instantiate runner object with processed config and register config into Registry class
        """
        self.config, self.duplicate_config = build_config(pArgs)
        self.config_registry()

    def config_registry(self):
        """
        Register config into Registry class
        """

        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)  # register logger path
        interface.World_param_Interface(self.config)
        if self.config['model'].get('graphic', False):
            param = Registry.mapping['world_mapping']['setting'].param
            if self.config['command']['world'] in ['cityflow', 'sumo']:
                roadnet_path = param['dir'] + param['roadnetFile']
            else:
                roadnet_path = param['road_file_addr']
            interface.Graph_World_Interface(roadnet_path)  # register graphic parameters in Registry class
        interface.Logger_path_Interface(self.config)
        # make output dir if not exist
        if not os.path.exists(Registry.mapping['logger_mapping']['path'].path):
            os.makedirs(Registry.mapping['logger_mapping']['path'].path)        
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)

    def run(self):
        logger = setup_logging(logging_level)
        self.trainer = Registry.mapping['trainer_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](logger)
        self.task = Registry.mapping['task_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](self.trainer)
        start_time = time.time()
        self.task.run()
        logger.info(f"Total time taken: {time.time() - start_time}")


if __name__ == '__main__':
    test = Runner(args)
    test.run()

# python run.py --task tsc --agent adapt_comm --world cityflow --network cityflow_grid4x4 --prefix my_adapt_exp
