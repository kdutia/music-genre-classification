import torch
import torch.nn as nn

from datasets import MelSpecDataset
import matplotlib.pyplot as plt

def test_dataset():

    dataset = MelSpecDataset(csv_loc="data/wav_files.csv", data_dir="data", transforms=None)

    fig = plt.figure()


    # show first 3 samples 
    firstn = 3
    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['image'].shape, sample['image'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample['image'])

        if i == firstn:
            plt.show()
            break

def test_dataloader():
    BATCH_SIZE = 5

    dataset = MelSpecDataset(csv_loc="data/wav_files.csv", data_dir="data", transforms=None)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    next_item = next(iter(data_loader))

    for i in range(BATCH_SIZE):
        ax = plt.subplot(BATCH_SIZE,1,i+1)
        plt.tight_layout()

        ax.set_title(next_item['label'][i])
        ax.axis('off')
        plt.imshow(next_item['image'][i,...])
    
    plt.show()

# RUN
test_dataloader()