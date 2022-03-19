def inspect_tflite(interpreter):
    print(interpreter)

    interpreter.allocate_tensors()

    print("== Input details ==")
    print("name:", interpreter.get_input_details()[0]['name'])
    print("shape:", interpreter.get_input_details()[0]['shape'])
    print("type:", interpreter.get_input_details()[0]['dtype'])

    print("\nDUMP INPUT")
    print(interpreter.get_input_details()[0])

    print("\n== Output details ==")
    print("name:", interpreter.get_output_details()[0]['name'])
    print("shape:", interpreter.get_output_details()[0]['shape'])
    print("type:", interpreter.get_output_details()[0]['dtype'])

    print("\nDUMP OUTPUT")
    print(interpreter.get_output_details()[0])
