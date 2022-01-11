import argparse

import torch
import model.detector
import utils.utils

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='', 
                        help='The path of the .pth model to be transformed')

    parser.add_argument('--output', type=str, default='./model.onnx', 
                        help='The path where the onnx model is saved')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, True, cfg=cfg).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    #sets the module in eval node
    model.eval()

    img_size = opt.imgsz
    # test_data = torch.rand(1, 3, cfg["height"], cfg["width"]).to(device)
    test_data = torch.rand(1, 3, img_size[0], img_size[1]).to(device)
    if opt.half:
        test_data = test_data.half()
        model.half()
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     opt.output,               # where to save the model (can be a file or file-like object)
                     input_names=["images"],
                     output_names=["output"],
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=11,          # the ONNX version to export the model to
                     do_constant_folding=True)  # whether to execute constant folding for optimization

    

