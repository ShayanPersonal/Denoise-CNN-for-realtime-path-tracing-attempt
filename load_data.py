import OpenEXR
import Imath
import random
import numpy as np
import matplotlib.pyplot as plt

def load_exr_data(filename, preprocess=False, concat=False, target=False):
  """Loads a multilayer OpenEXR file and returns a list of all features loaded as numpy arrays.
  You can use np.concatenate(x, 2) to create a 3D array of size [width, height, 14]"""
  infile = OpenEXR.InputFile(filename)
  color = get_layer(infile, 'Color')
  normal = get_layer(infile, 'Normal')
  albedo = get_layer(infile, 'Albedo')
  depth = get_layer(infile, 'Depth')
  color_var = get_layer(infile, 'ColorVar')
  normal_var = get_layer(infile, 'NormalVar')
  albedo_var = get_layer(infile, 'AlbedoVar')
  depth_var = get_layer(infile, 'DepthVar')

  # preprocess
  if preprocess:
    depth /= np.max(depth) + 0.00001
    depth_var /= np.max(depth_var) + 0.00001
    albedo_var /= np.max(albedo_var) + 0.00001
    color_var /= np.max(color_var) + 0.00001 
    normal_var /= np.max(normal_var) + 0.00001

  if not target:
    data = [color, normal, albedo, depth, color_var, normal_var, albedo_var, depth_var]
  else:
    data = [np.clip(color, 0, 1)]
  if concat:
    data = np.concatenate(data, axis=2)
    data = np.swapaxes(data, 0, 2)
    return data
  return data

def get_layer(infile, layer_name):
  """Returns a np array with all channels of a single layer in the EXR file with the matching layer name"""
  # extract channel names
  channel_names = []
  for layer in infile.header()['channels']:
    # add . to end of layer_name so we don't get layers that start with the same prefix too
    if layer_name+'.' in layer:
      channel_names.append(layer)
  # make sure we got something
  if not channel_names:
    print('Warning: Layer \'%s\' was not found.' % layer_name)
    return None
  # sort to RGB, XYZ, and remove A if more than one channel
  if channel_names:
    channel_names = sorted(channel_names)
    # if RGB, rearrange from BGR to RGB
    if channel_names[0].split('.')[-1] == 'B':
      channel_names = [channel_names[2], channel_names[1], channel_names[0]]
  # get image dimensions
  dw = infile.header()['dataWindow']
  size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
  # get data from channels
  pt = Imath.PixelType(Imath.PixelType.FLOAT)
  data = np.zeros((size[1], size[0], len(channel_names)))
  for i, name in enumerate(channel_names):
    data[:, :, i] = np.fromstring(infile.channel(name, pt), dtype=np.float32).reshape(size[1], size[0])
  return data

def save_exr_data(filename, data):
  """"""
  pass

def get_patches(filename, gt_filename, patch_size=64, num_patches=200, preprocess=False):
  data = load_exr_data(filename, preprocess=preprocess, concat=True)
  gt = load_exr_data(gt_filename, preprocess=preprocess)[0]
  w = data.shape[0]
  h = data.shape[1]

  candidate_patches = []
  candidate_patches_gt = []
  candidate_patch_scores = []
  total_score = 0
  for i in range(num_patches*4):
    # pick random point to be the top-left corner of patch
    x = random.randint(0, w-patch_size-1)
    y = random.randint(0, h-patch_size-1)
    patch = data[x:x+patch_size, y:y+patch_size, :]
    patch_gt = gt[x:x+patch_size, y:y+patch_size, :]
    score = get_score(patch)

    candidate_patches.append(patch)
    candidate_patches_gt.append(patch_gt)
    candidate_patch_scores.append(score)
    total_score += score

  patches = []
  patches_gt = []
  while len(patches) < num_patches:
    for i, patch in enumerate(candidate_patches):
      # probability of picking this patch is score/total_score
      p_pick = candidate_patch_scores[i] / total_score
      if random.random() < p_pick:
        # pick this patch and remove it from pool
        patches.append(patch)
        patches_gt.append(candidate_patches_gt[i])
        del candidate_patch_scores[i]
        del candidate_patches_gt[i]
        del candidate_patches[i]
        break

  return np.array(patches), np.array(patches_gt)

def get_score(patch):
  # score is total color variance + total normal variance
    return np.sum(patch[:, :, 10]) + np.sum(patch[:, :, 11])

if __name__ == "__main__":
  print("Testing load_exr_data")
  x = load_exr_data("training/data/7_train.exr", preprocess=True, concat=True, target=True)
  print(x.shape)
  plt.imshow(np.swapaxes(x, 0, 2))
  plt.show()