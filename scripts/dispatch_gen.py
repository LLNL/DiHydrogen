"""Generate runtime dispatch infrastructure for DiHydrogen."""

import itertools
import argparse
import re
import os.path



# Compute types present in H2.
# The index of each type in this list must match the index in the
# h2::ComputeTypes type list.
H2_COMPUTE_TYPES = ['float', 'double', 'std::int32_t', 'std::uint32_t']
# Number of bits per token.
H2_BITS_PER_COMPUTE_TYPE = (len(H2_COMPUTE_TYPES) - 1).bit_length()

# Regex for matching dispatch arguments.
# TODO: This will not handle any arguments that are string literals.
H2_ARG_REGEX = re.compile(r'"([a-zA-Z0-9_\[\]()<>{}&*:\' ]+)"')


def dispatch_table_str(
        name: str, device: str, entries: list[str], indent: int = 0) -> str:
    """Generate a dispatch table containing the entries."""
    indent_str = ' ' * indent
    table = indent_str + f'static std::array<::h2::internal::DispatchFunctionEntry, {len(entries)}> _dispatch_table_{name}_{device} = {{{{\n'
    for entry in entries:
        table += indent_str + f'{{{entry}}},\n'
    table += indent_str + '}};\n'
    return table


def dispatch_entry_str(
        impl_name: str, arg_strs: list[str], indent: int = 0) -> str:
    """Generate an entry for a dispatch table."""
    arg_list_str = ', '.join(arg_strs)
    indent_str = ' ' * indent
    entry = (
        f'{indent_str}reinterpret_cast<void*>(\n'
        f'{indent_str}    static_cast<void (*)({arg_list_str})>({impl_name})),\n'
        f'{indent_str}&::h2::internal::DispatchFunctionWrapper<void, {arg_list_str}>::call'
    )
    return entry


def do_dispatch_str(
        table_name: str,
        device: str,
        dispatch_name: str,
        dispatch_on: list[str],
        args: list[str],
        indent: int = 0) -> str:
    """Generate a do_dispatch call."""
    dispatch_args = ', '.join(dispatch_on)
    dispatch_on_str = f'::h2::DispatchOn<{len(dispatch_on)}>({dispatch_args})'
    args_str = ', '.join(args)
    indent_str = ' ' * indent
    dispatch_str = (
        f'{indent_str}::h2::do_dispatch(_dispatch_table_{table_name}_{device}, '
        f'"{dispatch_name}", {dispatch_on_str}, {args_str})'
    )
    return dispatch_str


def device_dispatch_str(
        get_device_str: str,
        table_name: str,
        dispatch_name: str,
        dispatch_on: list[str],
        cpu_args: list[str],
        gpu_args: list[str],
        indent: int = 0) -> str:
    """Generate a H2_DEVICE_DISPATCH dispatch block."""
    indent_str = ' ' * indent
    prefix_padding = ' ' * len('H2_DEVICE_DISPATCH(')
    cpu_dispatch = do_dispatch_str(
        table_name, 'cpu', dispatch_name + '_cpu', dispatch_on, cpu_args)
    gpu_dispatch = do_dispatch_str(
        table_name, 'gpu', dispatch_name + '_gpu', dispatch_on, gpu_args)
    dispatch_str = (
        f'{indent_str}H2_DEVICE_DISPATCH({get_device_str},\n'
        f'{indent_str}{prefix_padding}{cpu_dispatch},\n'
        f'{indent_str}{prefix_padding}{gpu_dispatch});\n'
    )
    return dispatch_str


def get_dispatch_types_in_order(num_types: int) -> list[tuple[str, ...]]:
    """Generate a list of dispatch type sets, in the order they should
    appear in the dispatch table."""
    return [types
            for types in itertools.product(*([H2_COMPUTE_TYPES]*num_types))]


def get_dispatch_token_for_type(type_name: str) -> int:
    """Return the dispatch token for a type."""
    return H2_COMPUTE_TYPES.index(type_name)


def get_dispatch_token_for_types(types: tuple[str, ...]) -> int:
    """Return the token used to dispatch on types."""
    types_token = 0
    shift_start = len(types) - 1
    for i, type_name in enumerate(types):
        t_token = get_dispatch_token_for_type(type_name)
        types_token |= t_token << H2_BITS_PER_COMPUTE_TYPE*(shift_start-i)
    return types_token


def get_dispatch_tokens_in_order(num_types: int) -> list[int]:
    """Generate a list of dispatch tokens in type order."""
    return [get_dispatch_token_for_types(types)
            for types in get_dispatch_types_in_order(num_types)]


def generate_dispatch_table(
        num_types: int,
        table_name: str,
        impl_name: str,
        device: str,
        arg_strs: list[str],
        indent: int = 0) -> str:
    """Generate a complete dispatch table.

    arg_strs and impl_name may contain format keys like '{T1}', ...,
    which will be replaced with the corresponding dispatch type.
    """
    entries = []
    for types in get_dispatch_types_in_order(num_types):
        format_dict = {f'T{i+1}': t for i, t in enumerate(types)}
        this_impl_name = impl_name.format(**format_dict)
        entry_args = [arg.format(**format_dict) for arg in arg_strs]
        entries.append(dispatch_entry_str(
            this_impl_name, entry_args, indent=2))
    return dispatch_table_str(table_name, device, entries, indent=indent)


def parse_name_line(line: str) -> str:
    """Return the base name for dispatch tables."""
    line = line.strip()
    name = line[len('// H2_DISPATCH_NAME: '):]
    return name


def parse_num_types_line(line: str) -> int:
    """Return the number of types to dispatch on."""
    line = line.strip()
    num_types = int(line[len('// H2_DISPATCH_NUM_TYPES: '):])
    return num_types


def parse_init_line(orig_line: str) -> tuple[str, str, list[str]]:
    """Extract table generation information (name, device, arguments)."""
    line = orig_line.strip()
    line = line[len('// H2_DISPATCH_INIT'):]
    if line.startswith(':'):
        device = 'none'
        line = line[len(': '):]
    elif line.startswith('_CPU:'):
        device = 'cpu'
        line = line[len('_CPU: '):]
    elif line.startswith('_GPU'):
        device = 'gpu'
        line = line[len('_GPU: '):]
    else:
        raise RuntimeError(f'Do not know how to parse "{orig_line}"')

    name = line[:line.index('(')]
    line = line[len(name) + 1:-1]
    args = H2_ARG_REGEX.findall(line)

    return name, device, args


def parse_get_device_line(line: str) -> str:
    """Extract the code to get the device to dispatch on."""
    line = line.strip()
    line = line[len('// H2_DISPATCH_GET_DEVICE: "'):-1]
    return line


def parse_dispatch_on_line(line: str) -> list[str]:
    """Extract the arguments to dispatch on."""
    line = line.strip()
    line = line[len('// H2_DISPATCH_ON: '):]
    args = H2_ARG_REGEX.findall(line)
    return args


def parse_dispatch_args_line(orig_line: str) -> dict[str, list[str]]:
    """Extract the arguments pass to the dispatched function."""
    line = orig_line.strip()
    line = line[len('// H2_DISPATCH_ARGS'):]
    if line.startswith(':'):
        device = 'none'
        line = line[len(': '):]
    elif line.startswith('_CPU:'):
        device = 'cpu'
        line = line[len('_CPU: '):]
    elif line.startswith('_GPU'):
        device = 'gpu'
        line = line[len('_GPU: '):]
    else:
        raise RuntimeError(f'Do not know how to parse "{orig_line}"')

    args = H2_ARG_REGEX.findall(line)
    return {device: args}


def process_file(infile: str, outfile: str) -> None:
    """Generate dispatch code for infile and write to outfile."""
    with open(infile, 'r') as f:
        source_lines = f.readlines()

    out_lines = []
    name = None
    num_types = None
    get_device = None
    dispatch_on_args = None
    dispatch_args = {}
    for line in source_lines:
        start = line.lstrip()
        if start.startswith('// H2_DISPATCH_NAME'):
            name = parse_name_line(line)
        elif start.startswith('// H2_DISPATCH_NUM_TYPES'):
            num_types = parse_num_types_line(line)
        elif start.startswith('// H2_DISPATCH_INIT'):
            impl_name, device, args = parse_init_line(line)
            if name is None or num_types is None:
                raise RuntimeError(
                    'Name or num types not specified before dispatch')
            indent = len(line) - len(start)
            dispatch_table = generate_dispatch_table(
                num_types,
                name,
                impl_name,
                device,
                args,
                indent=indent)
            if device == 'gpu':
                dispatch_table = ('#ifdef H2_HAS_GPU\n'
                                  + dispatch_table
                                  + '#endif  // H2_HAS_GPU\n')
            dispatch_table = ('// BEGIN GENERATED DISPATCH TABLE\n'
                              + dispatch_table
                              + '// END GENERATED DISPATCH TABLE\n')
            out_lines.append(dispatch_table)
        elif start.startswith('// H2_DISPATCH_GET_DEVICE'):
            get_device = parse_get_device_line(line)
        elif start.startswith('// H2_DISPATCH_ON'):
            dispatch_on_args = parse_dispatch_on_line(line)
            if len(dispatch_on_args) != num_types:
                raise RuntimeError(
                    'Dispatch generation error:'
                    f' expected number of types {num_types}'
                    ' does not match number of dispatch arguments:'
                    f' {len(dispatch_on_args)} ({dispatch_on_args})')
        elif start.startswith('// H2_DISPATCH_ARGS'):
            dispatch_args.update(parse_dispatch_args_line(line))
        elif start.startswith('// H2_DO_DISPATCH'):
            indent = len(line) - len(start)
            if len(dispatch_args) == 1:
                # No H2_DEVICE_DISPATCH.
                dispatch_str = do_dispatch_str(
                    name,
                    device,
                    name,
                    dispatch_on_args,
                    dispatch_args[device],
                    indent=indent)
                dispatch_str += ';\n'
            elif len(dispatch_args) > 1:
                if not get_device:
                    raise RuntimeError(
                        'Cannot dispatch to multiple devices without a GET_DEVICE')
                dispatch_str = device_dispatch_str(
                    get_device,
                    name,
                    name,
                    dispatch_on_args,
                    dispatch_args['cpu'],
                    dispatch_args['gpu'],
                    indent=indent)
            else:
                raise RuntimeError('No dispatch arguments found')
            dispatch_str = ('// BEGIN GENERATED DISPATCH CALL\n'
                            + dispatch_str
                            + '// END GENERATED DISPATCH CALL\n')
            out_lines.append(dispatch_str)
            # Clear things out.
            name = None
            num_types = None
            get_device = None
            dispatch_on_args = None
            dispatch_args = {}
        else:
            out_lines.append(line)

    with open(outfile, 'w') as f:
        f.writelines(out_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Post-process files and generate dispatch code')
    parser.add_argument('--infile', type=str, help='Input file to postprocess')
    parser.add_argument('--outfile', type=str, help='Output file')
    args = parser.parse_args()
    if not os.path.isfile(args.infile):
        raise ValueError(f'Input file {args.infile} does not exist')
    process_file(args.infile, args.outfile)
