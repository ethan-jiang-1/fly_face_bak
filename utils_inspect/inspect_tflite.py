def _dump_tflite_tensor(tensor_dp):
    print("", "index:  ", tensor_dp['index'])
    print("", "name:   ", tensor_dp['name'])
    print("", "shape:  ", tensor_dp['shape'])
    print("", "type:   ", tensor_dp['dtype'])
    print("", "shape_s:", tensor_dp["shape_signature"])

def dump_ipt_info(ipt):
    print("== Input details ==")
    inputs = ipt.get_input_details()
    for idx in range(len(inputs)):
        input = inputs[idx]
        print("DUMP INPUT", idx) # , input)
        _dump_tflite_tensor(input)
    print()

    print("== Output details ==")
    outputs = ipt.get_output_details()
    for idx in range(len(outputs)):
        output = outputs[idx]
        print("DUMP OUTPUT", idx) # , output)
        _dump_tflite_tensor(output)
    print()


def inspect_tflite(interpreter):
    print(interpreter)

    interpreter.allocate_tensors()

    dump_ipt_info(interpreter)

