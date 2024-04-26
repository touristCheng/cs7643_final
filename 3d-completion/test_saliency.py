import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from captum.attr import Saliency, IntegratedGradients, LayerActivation, LayerGradCam, LayerConductance, LayerGradientXActivation, InternalInfluence, LayerConductance
from captum.attr import *
from utils.vis_utils import plot_single_pcd, plot_map
from utils.train_utils import *
from dataset import ShapeNetH5
from torch.utils.tensorboard import SummaryWriter

def compute_attributions(algo, inputs, **kwargs):
    '''
    A common function for computing captum attributions
    '''
    return algo.attribute(inputs, **kwargs)

def test():
    dataset_test = ShapeNetH5(train=False, npoints=args.num_points)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    if args.save_vis:
        save_gt_path = os.path.join(log_dir, 'pics', 'gt')
        save_partial_path = os.path.join(log_dir, 'pics', 'partial')
        save_completion_path = os.path.join(log_dir, 'pics', 'completion')
        save_debugging_path = os.path.join(log_dir, 'pics', 'debugging')
        save_saliency_path = os.path.join(log_dir, 'pics', 'saliency')
        os.makedirs(save_gt_path, exist_ok=True)
        os.makedirs(save_partial_path, exist_ok=True)
        os.makedirs(save_completion_path, exist_ok=True)
        os.makedirs(save_saliency_path, exist_ok=True)
        os.makedirs(save_debugging_path, exist_ok=True)
        
    args.savepath = save_debugging_path
    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    test_loss_cat = torch.zeros([16, 4], dtype=torch.float32).cuda()
    cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 150 * 26
    novel_cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 50 * 26
    cat_num = torch.cat((cat_num, novel_cat_num), dim=0)
    cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft', 
                'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']
    #idx_to_plot = [i for i in range(0, 1600, 75)]
    # CHANGE: IDX TO PLOT 0 TO 5
    idx_to_plot = [0,1,2,3]

    logging.info('Testing...')
    
    fig = 2048
    cat_ready = []

    #visualizers = ["layeractivation", "saliency", "integratedgradients", "deeplift", "gradientshap"]
    visualizers = ["layeractivation", "saliency"]

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            #if i > fig:
            #  break

            #if i != fig:
            #  continue

            label, inputs_cpu, gt_cpu = data
            label_id = int(label[0])
            label_name = cat_name[label_id]

            
            if label_id >= 16:
              continue

            if label_name in cat_ready:
              continue
            else:
              cat_ready.append(label_name) 

            print("Data ID: ", i)
            print("Label: ", label_id, " name: ", label_name)

            inputs = inputs_cpu.float().cuda()
            gt = gt_cpu.float().cuda().requires_grad_(True)
            inputs = inputs.transpose(2, 1).contiguous().requires_grad_(True)

            #Layer Activation Decoder
            #print(net)
            if "layeractivation" in visualizers:
              layer_act = LayerActivation(net, net.module.decoder)
              attr_act = layer_act.attribute(inputs, additional_forward_args=(gt,))
              #print(attr_act)
              #print("Decoder layer activation shape: ", attr_act[0].shape)
              for j in range(args.batch_size):
                    #idx = i * args.batch_size + j
                    #if idx in idx_to_plot:
                    plot_single_pcd(attr_act[j].transpose(2,1)[0].cpu().numpy(), os.path.join(save_debugging_path, '%s_%d_layeractivation_decoder.png' % (label_name,j)))
            
            #Saliency
            if "saliency" in visualizers:
              layer_act = Saliency(net.module.encoder)
              input_embedding = net.module.encoder(inputs)
              saliency = torch.zeros_like(inputs)
              #for feature in range(input_embedding.shape[1]):
              for feature in range(input_embedding.shape[1]):
                #print("Calculating saliency for feature: ", feature)
                attr_act = layer_act.attribute(inputs,target=feature)
                #print(attr_act)
                saliency += attr_act
                #print("Encoder Saliency shape: ", attr_act[0].shape)
              for j in range(args.batch_size):
                    #idx = i * args.batch_size + j
                    #if idx in idx_to_plot:
                    plot_map(inputs_cpu[j].cpu().numpy(), saliency[j].transpose(1,0), os.path.join(save_saliency_path, '%s_%d_saliency_encoder.png' % (label_name,j)))

            #DeepLift
            if "deeplift" in visualizers:
              deeplift = torch.zeros_like(inputs)
              for feature in range(input_embedding.shape[1]):
                #print("Calculating deeplift for feature: ", feature)
                layer_act = DeepLift(net.module.encoder)
                baseline = torch.zeros_like(inputs)
                attr_act = layer_act.attribute(inputs,baselines=baseline,target=feature)
                #print(attr_act)
                deeplift += attr_act
              #print("Encoder Deeplift shape: ", attr_act[0].shape)
              for j in range(args.batch_size):
                    #idx = i * args.batch_size + j
                    #if idx in idx_to_plot:
                    #plot_single_pcd(deeplift[j].transpose(1,0).cpu().numpy(), os.path.join(save_debugging_path, '%d_deeplift_encoder.png' % j))
                    plot_map(inputs_cpu[j].cpu().numpy(), deeplift[j].transpose(1,0), os.path.join(save_debugging_path, '%s_%d_deeplift_encoder.png' % (label_name,j)))
            

            if "integratedgradients" in visualizers:
              integrated_gradients = torch.zeros_like(inputs)
              for feature in range(input_embedding.shape[1]):
              #for feature in range(0):
                #print("Calculating integrated gradients for feature: ", feature)
                layer_act = IntegratedGradients(net.module.encoder)
                baseline = torch.zeros_like(inputs)
                attr_act = layer_act.attribute(inputs,baselines=baseline,target=feature)
                #print(attr_act)
                integrated_gradients += attr_act
                #attr_act = layer_act.attribute(inputs,baselines=baseline,target=target)
              #print("Encoder Integrated Gradients shape: ", attr_act[0].shape)
              for j in range(args.batch_size):
                    #idx = i * args.batch_size + j
                    #if idx in idx_to_plot:
                    #plot_single_pcd(integrated_gradients[j].transpose(1,0).cpu().numpy(), os.path.join(save_debugging_path, '%d_integratedgradients_encoder.png' % j))
                    plot_map(inputs_cpu[j].cpu().numpy(), integrated_gradients[j].transpose(1,0), os.path.join(save_debugging_path, '%s_%d_integratedgradients_encoder.png' % (label_name,j)))

            #GradientShap
            if "gradientshap" in visualizers:
              gradientshap = torch.zeros_like(inputs)
              for feature in range(input_embedding.shape[1]):
                #print("Calculating gradientshap for feature: ", feature)
                layer_act = GradientShap(net.module.encoder)
                #input_embedding = net.module.encoder(inputs)
                baseline = torch.randn(inputs.shape).cuda()
                attr_act = layer_act.attribute(inputs,baselines=baseline,target=0)
                gradientshap += attr_act
              #print("Encoder GradientShap shape: ", attr_act[0].shape)
              for j in range(args.batch_size):
                    #idx = i * args.batch_size + j
                    #if idx in idx_to_plot:
                    #plot_single_pcd(attr_act[j].transpose(1,0).cpu().numpy(), os.path.join(save_debugging_path, '%d_gradientshap_encoder.png' % j))
                    plot_map(inputs_cpu[j].cpu().numpy(), gradientshap[j].transpose(1,0), os.path.join(save_debugging_path, '%s_%d_gradientshap_encoder.png' % (label_name,j)))

            result_dict = net(inputs,gt)

            if args.save_vis:
                for j in range(args.batch_size):
                    plot_single_pcd(inputs_cpu[j].cpu().numpy(), os.path.join(save_partial_path, '%s_%d_partial.png' % (label_name,j)))
                    plot_single_pcd(gt_cpu[j], os.path.join(save_gt_path, '%s_%d_reference_gt.png' % (label_name,j)))
                    plot_single_pcd(result_dict[j].cpu().numpy(), os.path.join(save_completion_path, '%s_%d_result.png' % (label_name,j)))

        logging.info('Loss per category:')
        category_log = ''
        for i in range(16):
            category_log += '\ncategory name: %s' % (cat_name[i])
            for ind, m in enumerate(metrics):
                scale_factor = 1 if m == 'f1' else 10000
                category_log += ' %s: %f' % (m, test_loss_cat[i, ind] / cat_num[i] * scale_factor)
        logging.info(category_log)

        logging.info('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        logging.info(overview_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test()
