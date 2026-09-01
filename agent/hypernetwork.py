import math

import torch
import torch.nn as nn


class GeneratedParamScaler:
    """
    Reset fan-in/out scaling for hypernetwork-generated target parameters.

    Hypernetworks emit unconstrained flattened parameters. This scaler maps each
    generated target layer back to a standard initialization scale before the
    generated weight/bias is used in the target network forward pass.
    """

    def __init__(
        self,
        enabled=False,
        mode='fan_in',
        hidden_gain=math.sqrt(2.0),
        output_gain=1.0,
        bias_scale=1.0,
    ):
        self.enabled = bool(enabled)
        self.mode = str(mode or 'fan_in').lower()
        self.hidden_gain = float(hidden_gain)
        self.output_gain = float(output_gain)
        self.bias_scale = float(bias_scale)

    def _gain(self, layer_idx, layer_count):
        return self.output_gain if layer_idx == layer_count - 1 else self.hidden_gain

    def _weight_scale(self, out_dim, in_dim, layer_idx, layer_count):
        if not self.enabled:
            return 1.0

        fan_in = max(1.0, float(in_dim))
        fan_out = max(1.0, float(out_dim))
        gain = self._gain(layer_idx, layer_count)

        if self.mode in ('fan_in', 'kaiming', 'kaiming_normal'):
            return gain / math.sqrt(fan_in)
        if self.mode in ('fan_out',):
            return gain / math.sqrt(fan_out)
        if self.mode in ('fan_avg', 'fan_in_out', 'xavier', 'xavier_normal'):
            return gain * math.sqrt(2.0 / (fan_in + fan_out))
        if self.mode in ('none', 'identity', 'off'):
            return 1.0
        raise ValueError(f"Unknown generated parameter RF mode: {self.mode}")

    def _bias_scale(self, in_dim, layer_idx, layer_count):
        if not self.enabled:
            return 1.0
        fan_in = max(1.0, float(in_dim))
        return self.bias_scale * self._gain(layer_idx, layer_count) / math.sqrt(fan_in)

    def scale_weight(self, weight, out_dim, in_dim, layer_idx, layer_count):
        return weight * self._weight_scale(out_dim, in_dim, layer_idx, layer_count)

    def scale_bias(self, bias, in_dim, layer_idx, layer_count):
        return bias * self._bias_scale(in_dim, layer_idx, layer_count)


def build_generated_param_scaler(
    cfg,
    output_gain_key=None,
    default_output_gain=1.0,
    enabled_key='hyper_rf_scaling',
):
    """
    Build runtime RF scaler from model config.
    """

    output_gain = cfg.get('hyper_rf_output_gain', default_output_gain)
    if output_gain_key is not None:
        output_gain = cfg.get(output_gain_key, output_gain)

    return GeneratedParamScaler(
        enabled=bool(cfg.get(enabled_key, False)),
        mode=cfg.get('hyper_rf_mode', 'fan_in'),
        hidden_gain=float(cfg.get('hyper_rf_hidden_gain', math.sqrt(2.0))),
        output_gain=float(output_gain),
        bias_scale=float(cfg.get('hyper_rf_bias_scale', 1.0)),
    )


def build_generated_param_init_config(
    cfg,
    output_gain_key=None,
    default_output_gain=1.0,
):
    """
    Build init-only RF config for layerwise hypernetwork output heads.

    This keeps the HyperMARL RF idea at initialization time instead of applying
    a repeated scale to every generated target parameter during forward passes.
    """

    output_gain = cfg.get('hyper_rf_output_gain', default_output_gain)
    if output_gain_key is not None:
        output_gain = cfg.get(output_gain_key, output_gain)

    return {
        'rf_init': bool(cfg.get('hyper_rf_init', cfg.get('hyper_rf_scaling', False))),
        'rf_mode': cfg.get('hyper_rf_mode', 'fan_in'),
        'rf_hidden_gain': float(cfg.get('hyper_rf_hidden_gain', math.sqrt(2.0))),
        'rf_output_gain': float(output_gain),
        'chunk_rf_mode': str(cfg.get('hyper_chunk_rf_mode', 'shared')).lower(),
    }


class LinearHyperNetwork(nn.Module):
    """
    Linear HyperMARL-style generator for flattened target-network parameters.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)


class MLPHyperNetwork(nn.Module):
    """
    MLP generator for flattened target-network parameters.

    With ``rf_init`` this head gets the same fan-in calibration the layerwise and
    chunked heads already had. Without it the output layer is initialized from
    its own fan-in (the hypernetwork hidden width) and knows nothing about the
    target layers it is emitting, so a 160-input critic layer and a 32-input
    actor layer come out at the same scale. That made ``hyper_rf_init`` a silent
    no-op for ``head_mode=flat``, which is exactly the all_weights baseline — so
    the flag could only ever be tested on the compressed heads.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        dropout=0.0,
        target_layout=None,
        rf_init=False,
        rf_hidden_gain=math.sqrt(2.0),
        rf_output_gain=1.0,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        if rf_init:
            if not target_layout:
                raise ValueError('rf_init on a flat head requires a target_layout')
            self._apply_rf_init(
                LayerwiseHyperNetwork._normalize_layout(target_layout),
                rf_hidden_gain,
                rf_output_gain,
            )

    def _apply_rf_init(self, layout, hidden_gain, output_gain):
        """Correctly scaled target init in the bias, small deviations in the weight.

        Same construction as ChunkedHyperNetwork._init_rf_generator, applied per
        target layer to the slice of the flat output the layer occupies.
        """
        output_layer = LayerwiseHyperNetwork._last_linear(self.net)
        layer_count = len(layout)
        with torch.no_grad():
            output_layer.bias.zero_()
            for layer_idx, (_, (out_dim, in_dim), weight_start, bias_start, bias_end) in enumerate(layout):
                gain = float(output_gain if layer_idx == layer_count - 1 else hidden_gain)
                target_weight = output_layer.weight.new_empty(out_dim, in_dim)
                nn.init.orthogonal_(target_weight, gain=gain)
                output_layer.bias[weight_start:bias_start].copy_(target_weight.reshape(-1))

                deviation = output_layer.weight.new_empty(
                    bias_end - weight_start, output_layer.weight.shape[1]
                )
                nn.init.orthogonal_(deviation, gain=gain / math.sqrt(max(1.0, float(in_dim))))
                output_layer.weight[weight_start:bias_end].copy_(deviation)

    def forward(self, x):
        return self.net(x)


class LayerwiseHyperNetwork(nn.Module):
    """
    HyperMARL-style generator with one output head per target-network layer.

    The public HyperMARL code generates target weights and biases with separate
    heads for each actor/critic layer. The returned tensor is still flattened so
    existing LibSignal forward code can slice it without knowing which generator
    implementation was used.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        target_layout,
        hypernet_type='mlp',
        dropout=0.0,
        use_bias=True,
        head_init_gain=1.0,
        rf_init=False,
        rf_mode='fan_in',
        rf_hidden_gain=math.sqrt(2.0),
        rf_output_gain=1.0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.target_layout = self._normalize_layout(target_layout)
        if len(self.target_layout) == 0:
            raise ValueError('LayerwiseHyperNetwork requires a target_layout')

        kind = str(hypernet_type or 'mlp').lower()
        if kind not in ('linear', 'lin', 'mlp', 'nonlinear'):
            raise ValueError(f"Unknown hypernetwork type: {hypernet_type}")
        self.hypernet_type = kind
        self.weight_heads = nn.ModuleList()
        self.bias_heads = nn.ModuleList()
        layer_count = len(self.target_layout)

        for layer_idx, (_, (out_dim, in_dim), _, weight_end, bias_end) in enumerate(self.target_layout):
            weight_dim = int(out_dim) * int(in_dim)
            bias_dim = int(out_dim)
            if kind in ('linear', 'lin'):
                weight_head = nn.Linear(input_dim, weight_dim, bias=use_bias)
                bias_head = nn.Linear(input_dim, bias_dim, bias=use_bias)
            else:
                weight_head = self._build_mlp_head(
                    input_dim,
                    hidden_dims,
                    weight_dim,
                    dropout,
                    use_bias,
                )
                bias_head = self._build_mlp_head(
                    input_dim,
                    hidden_dims,
                    bias_dim,
                    dropout,
                    use_bias,
                )
            init_gain = float(head_init_gain)
            if rf_init:
                init_gain = float(rf_output_gain if layer_idx == layer_count - 1 else rf_hidden_gain)
            self._init_weight_head(
                weight_head,
                init_gain,
                target_out_dim=out_dim,
                target_in_dim=in_dim,
                target_orthogonal=rf_init,
            )
            self._init_bias_head(bias_head)
            self.weight_heads.append(weight_head)
            self.bias_heads.append(bias_head)

        expected_dim = int(self.target_layout[-1][-1])
        if expected_dim != self.output_dim:
            raise ValueError(
                f"Layerwise layout output_dim mismatch: layout={expected_dim}, "
                f"output_dim={self.output_dim}"
            )

    @staticmethod
    def _normalize_layout(target_layout):
        layout = list(target_layout or [])
        if not layout:
            return []

        if len(layout[0]) == 4:
            layer_layout = []
            if len(layout) % 2 != 0:
                raise ValueError('Named parameter layout must contain weight/bias pairs')
            for idx in range(0, len(layout), 2):
                weight_name, weight_shape, weight_start, weight_end = layout[idx]
                bias_name, bias_shape, bias_start, bias_end = layout[idx + 1]
                if not str(weight_name).endswith('.weight') or not str(bias_name).endswith('.bias'):
                    raise ValueError('Named parameter layout must alternate weight then bias')
                if int(weight_end) != int(bias_start):
                    raise ValueError('Named parameter layout must be contiguous')
                out_dim, in_dim = tuple(weight_shape)
                layer_name = str(weight_name).rsplit('.', 1)[0]
                layer_layout.append(
                    (
                        layer_name,
                        (int(out_dim), int(in_dim)),
                        int(weight_start),
                        int(weight_end),
                        int(bias_end),
                    )
                )
            return layer_layout

        layer_layout = []
        for idx, item in enumerate(layout):
            if len(item) != 5:
                raise ValueError('Target layout entries must have 4 or 5 fields')

            if isinstance(item[1], (tuple, list)):
                name, shape, weight_start, bias_start, bias_end = item
                out_dim, in_dim = tuple(shape)
            else:
                out_dim, in_dim, weight_start, bias_start, bias_end = item
                name = f'layer{idx}'
            layer_layout.append(
                (
                    str(name),
                    (int(out_dim), int(in_dim)),
                    int(weight_start),
                    int(bias_start),
                    int(bias_end),
                )
            )
        return layer_layout

    @staticmethod
    def _build_mlp_head(input_dim, hidden_dims, output_dim, dropout, use_bias):
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for idx in range(len(dims) - 1):
            linear = nn.Linear(dims[idx], dims[idx + 1], bias=use_bias)
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim, bias=use_bias))
        return nn.Sequential(*layers)

    @staticmethod
    def _last_linear(module):
        if isinstance(module, nn.Linear):
            return module
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                return layer
        raise TypeError('Hypernetwork head does not contain a Linear layer')

    def _init_weight_head(
        self,
        module,
        gain,
        target_out_dim=None,
        target_in_dim=None,
        target_orthogonal=False,
    ):
        layer = self._last_linear(module)
        if target_orthogonal and target_out_dim is not None and target_in_dim is not None:
            self._init_target_orthogonal_head(layer, target_out_dim, target_in_dim, gain)
        else:
            nn.init.orthogonal_(layer.weight, gain=float(gain))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _init_target_orthogonal_head(layer, target_out_dim, target_in_dim, gain):
        weight_dim = int(target_out_dim) * int(target_in_dim)
        if layer.weight.shape[0] != weight_dim:
            nn.init.orthogonal_(layer.weight, gain=float(gain))
            return

        with torch.no_grad():
            for col_idx in range(layer.weight.shape[1]):
                target_weight = layer.weight.new_empty(int(target_out_dim), int(target_in_dim))
                nn.init.orthogonal_(target_weight, gain=float(gain))
                layer.weight[:, col_idx].copy_(target_weight.reshape(-1))

    def _init_bias_head(self, module):
        layer = self._last_linear(module)
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        chunks = []
        for weight_head, bias_head in zip(self.weight_heads, self.bias_heads):
            chunks.append(weight_head(x))
            chunks.append(bias_head(x))
        return torch.cat(chunks, dim=-1)


class ChunkedHyperNetwork(nn.Module):
    """
    Ha et al. style chunked generator for flattened target-network parameters.

    A flat head spends ``hidden_dim * N_theta`` parameters because it gives every
    generated scalar its own row. This generator still emits *every* target
    weight, but each target layer is produced in row-blocks by one shared
    generator conditioned on a learnable chunk embedding, so the cost drops to
    roughly ``hidden_dim * chunk_size * in_dim`` per layer.

    The generator is weight-shared across the chunks of a layer, which is a
    genuine inductive bias: the same function maps the meta vector to every
    row-block, differing only through the chunk code. It is far weaker than the
    bounded FiLM/head adapters, which cannot emit a full weight matrix at all.

    ``generator_hidden`` controls how the meta vector and the chunk code are
    combined. With the default 0 the generator is a single ``nn.Linear`` over
    ``cat([h, c_j])``, so every chunk emits ``W_h h + W_c c_j + b``: the
    agent-dependent term is *identical* in every chunk and the per-agent part of
    the target matrix is one row-block tiled down the matrix. A positive value
    inserts one hidden layer after the concatenation, which lets the chunk code
    and the meta vector interact and gives each chunk its own agent-dependent
    block. It is usually also *cheaper*, because the wide output head then reads
    from ``generator_hidden`` units instead of ``trunk_dim + chunk_embed_dim``.

    ``chunk_rf_mode`` picks which construction ``rf_init`` uses:

    ``shared``
        The historical one. One orthogonal ``chunk_size x in_dim`` block goes
        into the generator's output *bias*, which every chunk of the layer
        reads, so the generated target matrix starts as that single block tiled
        down the rows. The tie is only broken by the random chunk-code path, and
        the generated 128x160 critic layer starts with an effective rank near 13
        instead of 128 -- worse conditioning than the same head with rf_init off.
    ``per_chunk``
        Draws *one* orthogonal ``out_dim x in_dim`` init for the whole target
        layer -- the same draw the flat head would make -- slices it into
        row-blocks and routes block ``j`` through the chunk-code path, which
        already has the parameters to carry it. The chunk codes are initialized
        to an orthonormal basis so ``W_c c_j`` reproduces block ``j`` exactly.
        Costs nothing extra and needs ``chunk_embed_dim >= n_chunks`` (plus
        ``generator_hidden >= n_chunks`` when a hidden generator is used).
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        target_layout,
        hypernet_type='mlp',
        dropout=0.0,
        use_bias=True,
        head_init_gain=1.0,
        chunk_size=8,
        chunk_embed_dim=16,
        generator_hidden=0,
        rf_init=False,
        rf_mode='fan_in',
        rf_hidden_gain=math.sqrt(2.0),
        rf_output_gain=1.0,
        chunk_rf_mode='shared',
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.target_layout = LayerwiseHyperNetwork._normalize_layout(target_layout)
        if len(self.target_layout) == 0:
            raise ValueError('ChunkedHyperNetwork requires a target_layout')

        chunk_size = int(chunk_size)
        chunk_embed_dim = int(chunk_embed_dim)
        if chunk_size <= 0:
            raise ValueError('hyper_chunk_size must be positive')
        if chunk_embed_dim <= 0:
            raise ValueError('hyper_chunk_embed_dim must be positive')
        self.chunk_embed_dim = chunk_embed_dim

        generator_hidden = int(generator_hidden or 0)
        if generator_hidden < 0:
            raise ValueError('hyper_chunk_generator_hidden must be non-negative')
        self.generator_hidden = generator_hidden

        chunk_rf_mode = str(chunk_rf_mode or 'shared').lower()
        if chunk_rf_mode not in ('shared', 'per_chunk'):
            raise ValueError(f"Unknown hyper_chunk_rf_mode: {chunk_rf_mode}")
        if chunk_rf_mode == 'per_chunk' and not rf_init:
            raise ValueError(
                'hyper_chunk_rf_mode=per_chunk only describes how hyper_rf_init '
                'lays down the target-layer init, so it does nothing on its own: '
                'enable hyper_rf_init or set hyper_chunk_rf_mode back to shared'
            )
        self.chunk_rf_mode = chunk_rf_mode

        kind = str(hypernet_type or 'mlp').lower()
        if kind not in ('linear', 'lin', 'mlp', 'nonlinear'):
            raise ValueError(f"Unknown hypernetwork type: {hypernet_type}")
        self.hypernet_type = kind

        hidden = list(hidden_dims or [])
        if kind in ('linear', 'lin') or not hidden:
            self.trunk = nn.Identity()
            trunk_dim = int(input_dim)
        else:
            layers = []
            dims = [int(input_dim)] + [int(dim) for dim in hidden]
            for idx in range(len(dims) - 1):
                layers.append(nn.Linear(dims[idx], dims[idx + 1], bias=use_bias))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            self.trunk = nn.Sequential(*layers)
            trunk_dim = dims[-1]
        self.trunk_dim = trunk_dim

        self.chunk_embeddings = nn.ParameterList()
        self.generators = nn.ModuleList()
        self.chunk_specs = []
        layer_count = len(self.target_layout)

        for layer_idx, (_, (out_dim, in_dim), _, _, _) in enumerate(self.target_layout):
            out_dim = int(out_dim)
            in_dim = int(in_dim)
            rows = min(chunk_size, out_dim)
            n_chunks = (out_dim + rows - 1) // rows
            gen_out = rows * in_dim + rows

            embedding = nn.Parameter(torch.empty(n_chunks, chunk_embed_dim))
            nn.init.normal_(embedding, mean=0.0, std=1.0)
            generator = self._build_generator(trunk_dim + chunk_embed_dim, gen_out, use_bias)

            init_gain = float(head_init_gain)
            if rf_init:
                init_gain = float(
                    rf_output_gain if layer_idx == layer_count - 1 else rf_hidden_gain
                )
                if chunk_rf_mode == 'per_chunk':
                    self._init_rf_per_chunk(
                        embedding,
                        generator,
                        trunk_dim,
                        generator_hidden,
                        rows,
                        in_dim,
                        n_chunks,
                        init_gain,
                    )
                else:
                    self._init_rf_generator(generator, rows, in_dim, init_gain)

            self.chunk_embeddings.append(embedding)
            self.generators.append(generator)
            self.chunk_specs.append((out_dim, in_dim, rows, n_chunks))

        expected_dim = int(self.target_layout[-1][-1])
        if expected_dim != self.output_dim:
            raise ValueError(
                f"Chunked layout output_dim mismatch: layout={expected_dim}, "
                f"output_dim={self.output_dim}"
            )

    def _build_generator(self, in_dim, out_dim, use_bias):
        """One chunk generator: plain Linear, or Linear-ReLU-Linear when a hidden width is set."""
        if self.generator_hidden <= 0:
            return nn.Linear(in_dim, out_dim, bias=use_bias)

        hidden = nn.Linear(in_dim, self.generator_hidden, bias=use_bias)
        nn.init.orthogonal_(hidden.weight, gain=math.sqrt(2.0))
        if hidden.bias is not None:
            nn.init.zeros_(hidden.bias)
        return nn.Sequential(
            hidden,
            nn.ReLU(),
            nn.Linear(self.generator_hidden, out_dim, bias=use_bias),
        )

    @staticmethod
    def _init_rf_generator(generator, rows, in_dim, gain):
        """Put a correctly scaled target-layer init in the bias, deviations in the weight.

        The bias is shared by every chunk of the layer, so all ``n_chunks`` row
        blocks of the generated target matrix start out as the *same* block. See
        ``_init_rf_per_chunk`` for the variant that gives each chunk its own.
        """
        output_layer = LayerwiseHyperNetwork._last_linear(generator)
        with torch.no_grad():
            default_weight = output_layer.weight.new_empty(rows, in_dim)
            nn.init.orthogonal_(default_weight, gain=float(gain))
            nn.init.orthogonal_(
                output_layer.weight,
                gain=float(gain) / math.sqrt(max(1.0, float(in_dim))),
            )
            if output_layer.bias is not None:
                output_layer.bias.zero_()
                output_layer.bias[: rows * in_dim].copy_(default_weight.reshape(-1))

    @staticmethod
    def _init_rf_per_chunk(
        embedding, generator, trunk_dim, generator_hidden, rows, in_dim, n_chunks, gain
    ):
        """Give every chunk its own slice of one target-layer init, via the code path.

        ``_init_rf_generator`` has to put the init in the generator bias, which
        every chunk shares, so the generated matrix starts as one block tiled
        ``n_chunks`` times and only the random code path breaks the tie. Here the
        layer gets a single orthogonal ``(n_chunks * rows, in_dim)`` draw -- what
        a flat head would use for the same target layer -- and chunk ``j`` is
        handed row-block ``j`` through the parameters that already exist for the
        chunk code, so the fix costs nothing.

        The codes are set to an orthonormal basis ``C`` (rows ``c_j``) and the
        code columns of the output head to ``B C``, where column ``j`` of ``B``
        is block ``j``. Orthonormal rows give ``C c_j = e_j``, hence
        ``(B C) c_j = B e_j = block_j`` exactly. That needs ``n_chunks`` codes to
        be linearly independent, i.e. ``chunk_embed_dim >= n_chunks``: the same
        bound docs/CHUNK_SIZE_AND_EMBED_DIM.md calls the saturation point of the
        embedding, spent here instead of left idle.

        With a hidden generator the code never reaches the output head directly,
        so the first ``n_chunks`` hidden units are reserved as chunk gates: unit
        ``j`` reads ``c_j`` and nothing else, so it fires at exactly 1 on chunk
        ``j`` and 0 elsewhere, and the output head maps it to block ``j``. The
        remaining units keep their normal init and carry the agent-dependent
        part.
        """
        embed_dim = int(embedding.shape[1])
        if embed_dim < n_chunks:
            raise ValueError(
                f'hyper_chunk_rf_mode=per_chunk needs chunk_embed_dim >= n_chunks, '
                f'got chunk_embed_dim={embed_dim} for a layer split into '
                f'{n_chunks} chunks. Raise hyper_chunk_embed_dim to at least '
                f'{n_chunks}, or use a larger chunk size.'
            )
        if generator_hidden and generator_hidden < n_chunks:
            raise ValueError(
                f'hyper_chunk_rf_mode=per_chunk reserves one hidden unit per chunk, '
                f'so it needs hyper_chunk_generator_hidden >= n_chunks, got '
                f'{generator_hidden} for a layer split into {n_chunks} chunks.'
            )

        output_layer = LayerwiseHyperNetwork._last_linear(generator)
        weight_numel = int(rows) * int(in_dim)
        deviation_gain = float(gain) / math.sqrt(max(1.0, float(in_dim)))

        with torch.no_grad():
            codes = embedding.new_empty(n_chunks, embed_dim)
            nn.init.orthogonal_(codes)
            embedding.copy_(codes)

            layer_init = output_layer.weight.new_empty(n_chunks * int(rows), int(in_dim))
            nn.init.orthogonal_(layer_init, gain=float(gain))
            # column j is block j, flattened the way forward() reads it back
            blocks = layer_init.reshape(n_chunks, weight_numel).t()

            if output_layer.bias is not None:
                output_layer.bias.zero_()

            if generator_hidden:
                hidden_layer = generator[0]
                hidden_layer.weight[:n_chunks, :trunk_dim].zero_()
                hidden_layer.weight[:n_chunks, trunk_dim:].copy_(codes)
                if hidden_layer.bias is not None:
                    hidden_layer.bias[:n_chunks].zero_()

                output_layer.weight[:, :n_chunks].zero_()
                output_layer.weight[:weight_numel, :n_chunks].copy_(blocks)
                free = output_layer.weight.shape[1] - n_chunks
                deviation = output_layer.weight.new_empty(output_layer.weight.shape[0], free)
                nn.init.orthogonal_(deviation, gain=deviation_gain)
                output_layer.weight[:, n_chunks:].copy_(deviation)
                return

            deviation = output_layer.weight.new_empty(output_layer.weight.shape[0], trunk_dim)
            nn.init.orthogonal_(deviation, gain=deviation_gain)
            output_layer.weight[:, :trunk_dim].copy_(deviation)

            code_weight = output_layer.weight.new_zeros(
                output_layer.weight.shape[0], embed_dim
            )
            code_weight[:weight_numel].copy_(blocks @ codes)
            output_layer.weight[:, trunk_dim:].copy_(code_weight)

    def forward(self, x):
        hidden = self.trunk(x)
        chunks = []
        for embedding, generator, (out_dim, in_dim, rows, n_chunks) in zip(
            self.chunk_embeddings, self.generators, self.chunk_specs
        ):
            expanded = hidden.unsqueeze(-2).expand(*hidden.shape[:-1], n_chunks, self.trunk_dim)
            codes = embedding.expand(*hidden.shape[:-1], n_chunks, self.chunk_embed_dim)
            generated = generator(torch.cat([expanded, codes], dim=-1))

            weight_part = generated[..., : rows * in_dim]
            bias_part = generated[..., rows * in_dim :]

            weight = weight_part.reshape(*hidden.shape[:-1], n_chunks * rows, in_dim)
            weight = weight[..., :out_dim, :].reshape(*hidden.shape[:-1], out_dim * in_dim)
            bias = bias_part.reshape(*hidden.shape[:-1], n_chunks * rows)[..., :out_dim]

            chunks.append(weight)
            chunks.append(bias)
        return torch.cat(chunks, dim=-1)


class PrototypeHyperNetwork(nn.Module):
    """K-prototype factorization of any generator head.

    Every conditioning result in this study has been read off one of two
    cardinalities. ``learned`` gives each intersection its own free code, which
    is a per-index table and therefore cannot cross a network. ``constmeta``
    gives every intersection the same code, and (q) measured that losing to a
    plain shared MLP by 61.7 s on Ingolstadt21 -- generating N copies of one
    policy is a redundant parameterization of a shared one. Nothing between the
    two has been run, and (q)'s own diagnosis of why constmeta fails is exactly
    an argument for the middle.

    So: K learned prototype codes, and each intersection's parameters are a
    convex combination of the K generated parameter sets,

        theta_i = sum_k alpha_ik * inner(p_k),   alpha_i = softmax(gate(meta_i) / T)

    The generator sees only the K codes, never the meta, so it runs K times per
    forward instead of once per (batch element, agent). p_k and the gate are
    both shaped independently of N, so the whole path still transfers.

    The two endpoints reproduce the two known results, which is what makes this
    checkable: K=1 is constmeta by construction, and K=N with a one-hot gate is
    the same function class as learned.
    """

    def __init__(self, inner, input_dim, output_dim, num_prototypes,
                 gate_hidden=64, temperature=1.0, gate_frozen=False):
        super().__init__()
        if num_prototypes < 1:
            raise ValueError('num_prototypes must be >= 1')
        self.inner = inner
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_prototypes = int(num_prototypes)

        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, self.input_dim))
        # Orthogonal rows so the K codes do not start near each other; two codes
        # that begin close generate near-identical parameters and collapse
        # together before the gate has learned to tell them apart.
        nn.init.orthogonal_(self.prototypes, gain=1.0)

        gate_hidden = int(gate_hidden)
        if gate_hidden > 0:
            self.gate = nn.Sequential(
                nn.Linear(self.input_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, self.num_prototypes),
            )
        else:
            self.gate = nn.Sequential(nn.Linear(self.input_dim, self.num_prototypes))
        if gate_frozen:
            # The C3 control: a fixed random partition. Answers whether the win
            # is "K sets of weights" (capacity) or "structurally similar
            # intersections sharing a policy" (the actual hypothesis).
            for param in self.gate.parameters():
                param.requires_grad_(False)
        self.gate_frozen = bool(gate_frozen)

        # A buffer, not an attribute, so an annealed temperature is carried by
        # the checkpoint and a resumed run does not silently restart the anneal.
        self.register_buffer('temperature', torch.tensor(float(temperature)))
        self.register_buffer('usage', torch.zeros(self.num_prototypes))

    def set_temperature(self, value):
        with torch.no_grad():
            self.temperature.fill_(max(1e-3, float(value)))

    def mixing_weights(self, meta):
        logits = self.gate(meta)
        return torch.softmax(logits / self.temperature.clamp_min(1e-3), dim=-1)

    def forward(self, meta):
        # [K, output_dim] -- independent of batch size and of agent count, which
        # is the whole compute argument: the flat head emits one parameter set
        # per (batch element, agent) and uses each exactly once.
        deltas = self.inner(self.prototypes)
        alpha = self.mixing_weights(meta)
        if self.training:
            with torch.no_grad():
                flat = alpha.reshape(-1, self.num_prototypes)
                self.usage.mul_(0.99).add_(0.01 * flat.mean(dim=0))
        return alpha @ deltas

    def diagnostics(self):
        """Prototype occupancy. Collapse to one active prototype means this has
        silently become constmeta, which (q) showed is worse than no
        hypernetwork at all -- so it has to be visible in the log."""
        usage = self.usage.detach()
        total = usage.sum().clamp_min(1e-8)
        share = usage / total
        entropy = -(share * share.clamp_min(1e-8).log()).sum()
        return {
            'proto_active': float((share > 0.01).sum().item()),
            'proto_entropy': float(entropy.item()),
            'proto_max_share': float(share.max().item()),
            'proto_temperature': float(self.temperature.item()),
        }

    @torch.no_grad()
    def reinit_dead(self, threshold=0.01):
        """Re-draw prototypes no intersection is using.

        A dead prototype receives no gradient through ``alpha @ deltas``, so it
        stays dead once it gets there; K then quietly falls below the K being
        reported. Re-drawing keeps the arm honest about its own cardinality.
        """
        usage = self.usage
        share = usage / usage.sum().clamp_min(1e-8)
        dead = (share <= threshold).nonzero(as_tuple=True)[0]
        if dead.numel() == 0 or dead.numel() == self.num_prototypes:
            return 0
        fresh = torch.empty(dead.numel(), self.input_dim, device=self.prototypes.device)
        nn.init.orthogonal_(fresh, gain=1.0)
        self.prototypes[dead] = fresh
        self.usage[dead] = usage.mean()
        return int(dead.numel())


class HyperNetwork(MLPHyperNetwork):
    """
    Backward-compatible name for the original MLP hypernetwork.
    """


def build_hypernetwork(
    hypernet_type,
    input_dim,
    hidden_dims,
    output_dim,
    dropout=0.0,
    target_layout=None,
    head_mode='flat',
    use_bias=True,
    head_init_gain=1.0,
    rf_init=False,
    rf_mode='fan_in',
    rf_hidden_gain=math.sqrt(2.0),
    rf_output_gain=1.0,
    chunk_size=8,
    chunk_embed_dim=16,
    chunk_generator_hidden=0,
    chunk_rf_mode='shared',
    num_prototypes=0,
    prototype_gate_hidden=64,
    prototype_temperature=1.0,
    prototype_gate_frozen=False,
):
    """
    Build a linear or MLP hypernetwork from config.

    ``chunk_*`` arguments only reach the chunked head; the other head modes
    ignore them the way they always have.

    ``num_prototypes`` of 0 returns the head unchanged and constructs nothing,
    so every existing configuration keeps its exact parameter initialization
    order and stays byte-reproducible. A positive value wraps whichever head
    was selected in the K-prototype factorization, which is orthogonal to the
    head mode -- flat, layerwise and chunked can each be factorized.
    """

    kind = str(hypernet_type or 'mlp').lower()
    mode = str(head_mode or 'flat').lower()

    if num_prototypes and int(num_prototypes) > 0:
        inner = build_hypernetwork(
            hypernet_type, input_dim, hidden_dims, output_dim,
            dropout=dropout, target_layout=target_layout, head_mode=head_mode,
            use_bias=use_bias, head_init_gain=head_init_gain, rf_init=rf_init,
            rf_mode=rf_mode, rf_hidden_gain=rf_hidden_gain,
            rf_output_gain=rf_output_gain, chunk_size=chunk_size,
            chunk_embed_dim=chunk_embed_dim,
            chunk_generator_hidden=chunk_generator_hidden,
            chunk_rf_mode=chunk_rf_mode,
        )
        return PrototypeHyperNetwork(
            inner,
            input_dim,
            output_dim,
            int(num_prototypes),
            gate_hidden=prototype_gate_hidden,
            temperature=prototype_temperature,
            gate_frozen=prototype_gate_frozen,
        )
    if mode in ('chunked', 'chunk', 'factorized'):
        return ChunkedHyperNetwork(
            input_dim,
            hidden_dims,
            output_dim,
            target_layout=target_layout,
            hypernet_type=kind,
            dropout=dropout,
            use_bias=use_bias,
            head_init_gain=head_init_gain,
            chunk_size=chunk_size,
            chunk_embed_dim=chunk_embed_dim,
            generator_hidden=chunk_generator_hidden,
            rf_init=rf_init,
            rf_mode=rf_mode,
            rf_hidden_gain=rf_hidden_gain,
            rf_output_gain=rf_output_gain,
            chunk_rf_mode=chunk_rf_mode,
        )
    if mode in ('layerwise', 'headed', 'official'):
        return LayerwiseHyperNetwork(
            input_dim,
            hidden_dims,
            output_dim,
            target_layout=target_layout,
            hypernet_type=kind,
            dropout=dropout,
            use_bias=use_bias,
            head_init_gain=head_init_gain,
            rf_init=rf_init,
            rf_mode=rf_mode,
            rf_hidden_gain=rf_hidden_gain,
            rf_output_gain=rf_output_gain,
        )
    if kind in ('linear', 'lin'):
        return LinearHyperNetwork(input_dim, output_dim)
    if kind in ('mlp', 'nonlinear'):
        return MLPHyperNetwork(
            input_dim,
            hidden_dims,
            output_dim,
            dropout=dropout,
            target_layout=target_layout,
            rf_init=rf_init,
            rf_hidden_gain=rf_hidden_gain,
            rf_output_gain=rf_output_gain,
        )
    raise ValueError(f"Unknown hypernetwork type: {hypernet_type}")
