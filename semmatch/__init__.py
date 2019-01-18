import argparse
from semmatch import commands, data
from semmatch.utils import register


_MAJOR = 0
_MINOR = 1
__version__ = "{0}.{1}".format(_MAJOR, _MINOR)


def main(prog):
    parser = argparse.ArgumentParser(description="Run SemMatch", usage='%(prog)s', prog=prog)
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    subparsers = parser.add_subparsers(title='commands', metavar='')
    command_names = register.list_available('command')
    for command_name in command_names:
        register.get_by_name('command', command_name).add_subparser(subparsers)

    args = parser.parse_args()

    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()