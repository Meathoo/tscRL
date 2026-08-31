"""Generate runner_hyperlight.py / driver_hyperlight.py inside a Unicorn clone.

Unicorn already ships its scaffolding twice -- runner_unicorn/driver_unicorn and
runner_heterolight/driver_heterolight are the same files with the model swapped
-- so adding a third method is a supported shape, not a fork. This script makes
that third copy from the heterolight pair rather than vendoring 500 lines of
someone else's MIT code into our repo: rerun it after any `git pull` in the
clone and the port follows their changes.

Run it from the root of the Unicorn checkout, with models/HyperLight.py and
structural_meta.py already in place:

    python install.py

Arms are selected at runtime, not here:

    HYPER_META_MODE=structural|learned|constant  python driver_hyperlight.py
    HYPER_STRUCT_SHRINK=0.38                     (optional, structural only)
"""

import os
import re
import sys

RUNNER_SRC, RUNNER_DST = 'runner_heterolight.py', 'runner_hyperlight.py'
DRIVER_SRC, DRIVER_DST = 'driver_heterolight.py', 'driver_hyperlight.py'

# Inserted just above each model construction. The meta table is built once from
# the env the caller already has, in env.tls_list order, which is the agent order
# the model indexes with.
META_BLOCK = '''{indent}# --- HyperLight port: per-intersection conditioning code ---
{indent}_meta_mode = os.environ.get('HYPER_META_MODE', 'structural')
{indent}_meta_table, _meta_raw = build_meta_table(
{indent}    {env}, shrink=float(os.environ.get('HYPER_STRUCT_SHRINK', 1.0)))
{indent}{log}
'''

LOG_ONCE = "print('[HyperLight] meta_mode=%s  %s' % (_meta_mode, summarize(_meta_raw)))"


def patch(src, dst, env_expr, model_var, agent_dim_expr, log):
    with open(src, 'r', encoding='utf-8') as handle:
        text = handle.read()

    text = text.replace('from models.HeteroLight import HeteroLight',
                        'import os\n'
                        'from models.HyperLight import HyperLight\n'
                        'from structural_meta import build_meta_table, summarize')
    text = text.replace('from runner_heterolight import Runner',
                        'from runner_hyperlight import Runner')

    # The construction call spans four lines and differs only in indentation and
    # argument expressions between the two files, so match it structurally.
    pattern = re.compile(
        r'^(?P<indent>[ \t]*)' + re.escape(model_var) + r' = HeteroLight\((?P<args>.*?)\)\.to\(',
        re.S | re.M)
    match = pattern.search(text)
    if match is None:
        raise SystemExit('could not find the HeteroLight construction in %s' % src)

    indent = match.group('indent')
    args = match.group('args')
    new_call = (indent + model_var + ' = HyperLight(' + args
                + ',\n' + indent + ' ' * (len(model_var) + 16) + 'meta_table=_meta_table'
                + ',\n' + indent + ' ' * (len(model_var) + 16) + 'meta_mode=_meta_mode).to(')
    block = META_BLOCK.format(indent=indent, env=env_expr,
                              log=(log if log else 'pass'))
    text = text[:match.start()] + block + new_call + text[match.end():]

    with open(dst, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print('wrote %s' % dst)


def main():
    for required in (RUNNER_SRC, DRIVER_SRC, 'models/HyperLight.py', 'structural_meta.py'):
        if not os.path.exists(required):
            raise SystemExit('missing %s -- run this from the Unicorn checkout root' % required)

    # Only the driver logs the feature summary: the runners are ray actors and
    # NUM_META_AGENTS copies of the same line is noise.
    patch(RUNNER_SRC, RUNNER_DST, 'self.env', 'self.local_network',
          'self.env.tls_agent_space', log='')
    patch(DRIVER_SRC, DRIVER_DST, 'global_env', 'global_network',
          'global_env.tls_all_agent_space', log=LOG_ONCE)
    print('done; run with HYPER_META_MODE=structural python %s' % DRIVER_DST)


if __name__ == '__main__':
    sys.exit(main())
