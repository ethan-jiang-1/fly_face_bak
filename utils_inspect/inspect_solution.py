
DEBUG = False

def decode_binarypb(binarypb):
    from mediapipe.framework import calculator_pb2
    from mediapipe.python._framework_bindings import calculator_graph

    canonical_graph_config_proto = calculator_pb2.CalculatorGraphConfig()
    canonical_graph_config_proto.ParseFromString(binarypb)

    graph = calculator_graph.CalculatorGraph(graph_config=canonical_graph_config_proto)
    return graph.text_config

def _debug_config(solution):
    config_bin = solution._graph.binary_config
    config_txt = solution._graph.text_config
    print("\nconfig_bin {}\n".format(len(config_bin)), config_bin)
    print()
    print("\nconfig_txt {}\n".format(len(config_txt)), config_txt)
    print()

    decodedpb = decode_binarypb(config_bin)
    print("\ndecondedpb {}\n".format(len(decodedpb)), decodedpb)
    print()


def inspect_solution(solution):
    print(solution)
    
    if DEBUG:
        _debug_config(solution)

