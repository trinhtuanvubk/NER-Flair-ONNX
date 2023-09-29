import export_onnx
import args

def main():
    opt = args.get_args()
    method = getattr(export_onnx, opt.scenario)
    method(opt)


if __name__ == "__main__":
    main()