def test_dataset():
    from datasets import MelSpecDataset
    import matplotlib.pyplot as plt

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

# RUN
test_dataset()