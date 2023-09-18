import ner_onnx
import util

def main():
    args = util.get_args()
    method = getattr(ner_onnx, args.scenario)
    method(args)


if __name__ == "__main__":
    main()