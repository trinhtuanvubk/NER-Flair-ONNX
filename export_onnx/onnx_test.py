import onnxruntime

def onnx_test(args):
    # run onnx session
    sess = onnxruntime.InferenceSession(args.embedding_path, providers=['CUDAExecutionProvider'])
    
    print("Embedding Information Test")
    # input info
    print([x.name for x in sess.get_inputs()])
    print([x.shape for x in sess.get_inputs()])

    # output infro
    print([x.name for x in sess.get_outputs()])
    print([x.shape for x in sess.get_outputs()])



    sess = onnxruntime.InferenceSession(args.tagger_path, providers=['CUDAExecutionProvider'])
    
    print("Sequence Tagger Information Test")
    # input info
    print([x.name for x in sess.get_inputs()])
    print([x.shape for x in sess.get_inputs()])

    # output infro
    print([x.name for x in sess.get_outputs()])
    print([x.shape for x in sess.get_outputs()])
