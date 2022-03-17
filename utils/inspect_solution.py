def inspect_solution(solution):
    print(solution)
    config_bin = solution._graph.binary_config
    config_txt = solution._graph.text_config
    print("\nconfig_bin\n", config_bin)
    print()
    print("\nconfig_txt\n", config_txt)
    print()
