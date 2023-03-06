from types import SimpleNamespace
from utils.training_utils import *
from models.bidirectional_transformer import BidirectionalTransformer
from utils.shape_utils import *
from utils.generate_shapes import generate_blob, generate_line
from torch import nn
import argparse
import os

# Arg parsing
# Add default parameters
default_args = SimpleNamespace()
# Training parameters
default_args.name = "DEFAULT_NAME"
default_args.epochs = 2000
default_args.start_from_epoch = 0
default_args.learning_rate = 1e-6
default_args.accum_grad = 10
default_args.psz = 16
default_args.in_dim = default_args.psz ** 2
default_args.dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Transformer parameters
default_args.num_image_tokens = 24
default_args.num_unmask_tokens = int(default_args.num_image_tokens*.5)
default_args.num_mask_tokens = 20 #default_args.num_image_tokens - default_args.num_unmask_token
default_args.dim = 768
default_args.hidden_dim = 3072
default_args.n_layers = 20
#Image parameters
default_args.H = 256
default_args.W = 256
default_args.shape = "Blob" #"Blob", "Line"
default_args.use_slope = False
default_args.normalize_embed = False
default_args.scale_embed = 1.0e0
# Instantiate the parser
parser = argparse.ArgumentParser()
for key, val in default_args.__dict__.items():
    parser.add_argument(f"--{key}", type=type(val), default=val)


class TrainTransformer:
    def __init__(self, ims, args):
        self.model = BidirectionalTransformer(args).to(device=args.dev)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, betas=(0.9, 0.96),
                                      weight_decay=4.5e-2)
        self.ims = ims
        self.loss_list = []
        self.mask_ratio_list = []
        self.accuracy_list = []
        self.train_dataset = None

    def train(self, args):
        train_dataset = load_data(self.ims, args)
        self.train_dataset = train_dataset
        len_train_dataset = len(train_dataset)
        shuffle_set = np.arange(0, len_train_dataset)
        step = 0
        for epoch in range(1, args.epochs + 1):
            np.random.shuffle(shuffle_set)
            epoch_loss = 0
            epoch_accuracy = []
            for idx in shuffle_set:
                img = train_dataset[idx][0]
                pts = train_dataset[idx][1].cpu().numpy()
                img = img.to(device=args.dev)
                for _ in range(5):
                    # Generate a masked region
                    _, mask_corners = gen_mask_rect(pts)
                    # get patches from the unmasked region
                    unmask_patches, unmask_coords = get_unmasked_aligned_blob_points(img, pts, mask_corners,
                                                                                     int(args.num_unmask_tokens), args)
                    #get patches from masked region
                    mask_patches, mask_coords = get_random_mask_points(img, mask_corners, int(args.num_mask_tokens),
                                                                       args)
                    #normalize mask coordinates
                    mask_coords = torch.divide(mask_coords, args.H)
                    unmask_coords = torch.divide(unmask_coords, args.H)

                    #Run the model and get predictions for each pixel
                    preds = self.model(unmask_patches, unmask_coords, mask_coords)
                    preds = preds.squeeze()
                    #Get target colors
                    target = mask_patches.to(args.dev).squeeze()

                    # loss = F.cross_entropy((logits.T*mask_idxs).T.reshape(-1, logits.size(-1)), (target*mask_idxs).reshape(-1))
                    # loss = ((preds[mask_idxs,...] - target[mask_idxs,...])**2).mean(dim=-1) #Patchwise loss

                    lossfn = nn.BCELoss()
                    loss = lossfn(preds, target)
                    # print(loss.cpu().detach().numpy().item())

                    # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                    epoch_loss += loss.cpu().detach().numpy().item()
                    # add an accuracy count here

                    epoch_accuracy.append(100 * np.mean(
                        ((preds > .5).to(torch.int) == (target > .5).to(torch.int)).to(torch.int).cpu().numpy()))
                    loss.backward()
                    if step % args.accum_grad == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                    step += 1

            epoch_accuracy = np.mean(epoch_accuracy)
            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Epoch Loss = {np.round(epoch_loss, 4)}, Epoch accuracy = {np.round([epoch_accuracy], 4)[0]}")
            if epoch % 100 == 0:
                #MAKE SAVING SCRIPT HERE
                try:
                    os.mkdir("./ckpts")
                    print('Checkpoints created at ./ckpts')
                except:
                    print("ckpts already created")

                try:
                    os.mkdir(f'./ckpts/{args.name}')
                    print(f'Run directory created at ./ckpts/{args.name}')
                except:
                    print(f'Run directory at ./ckpts/{args.name} already exists! Did you forget to name the run?')
                torch.save(self.model.state_dict(), f'./ckpts/{args.name}/{epoch}.pt')
                print(f'State Saved at ./ckpts/{args.name}/{epoch}.pt')
            self.loss_list.append(np.round(epoch_loss, 4))
            self.accuracy_list.append(np.round(epoch_accuracy, 4))

if __name__ == '__main__':
    args = parser.parse_args()
    NUM_TRAINING_SAMPLES = 100
    training_list = []
    for _ in range(NUM_TRAINING_SAMPLES):
        if args.shape == "Blob":
            img, rad = generate_blob(args.H, args.W)
        elif args.shape == "Line":
            img, rad = generate_line(args.H,args.W,args.use_slope)
        img = img.squeeze()
        pts = get_pts_on_curve(rad, P=500).squeeze()
        training_list.append((img, pts))
    trained_model = TrainTransformer(training_list, args)
    trained_model.train(args)
    np.save(f'./ckpts/{args.name}/accuracy', trained_model.accuracy_list)
    np.save(f'./ckpts/{args.name}/loss', trained_model.loss_list)