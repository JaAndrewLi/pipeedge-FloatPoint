import numpy as np
import matplotlib.pyplot as plt
import os

batch_num = 10
layer_num = 48

for batch in range(1, batch_num + 1):
    print(f"In batch {batch}")
    fig, axes = plt.subplots(6, 8, figsize=(30, 20))

    for layer in range(1, layer_num + 1):
        file_name = f"original_data/batch_{batch}/layer_{layer}_x.npy"
        data = np.load(file_name).flatten()

        # hist_data = np.histogram(data, bins=100)
        row = (layer - 1) // 8
        col = (layer - 1) % 8

        # axes[row, col].bar(hist_data[1][:-1], hist_data[0])
        axes[row, col].hist(data, bins=100)
        axes[row, col].set_title(f'layer_{layer}, batch_{batch}')
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Frequency')
        # axes[row, col].set_yscale('log')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"original_data_plot/batch_{batch}.png")
    plt.close()


# batch_num = 1
# layer_num = 48

# for batch in range(1, batch_num + 1):
#     print(f"In batch {batch}")
#     # os.makedirs(f"original_data_plot/batch_{batch}", exist_ok=True)
    
#     for layer in range(1, layer_num + 1):
#         file_name = f"original_data/batch_{batch}/layer_{layer}_x.npy"
#         data = np.load(file_name).flatten()

#         # hist_data = np.histogram(data, bins=100)

#         # plt.figure(figsize=(10, 6))
#         # plt.bar(hist_data[1][:-1], hist_data[0])
#         plt.hist(data, bins=500)
#         plt.title(f'Data Distribution: layer_{layer}, batch_{batch}')
#         plt.xlabel('Value')
#         plt.ylabel('Frequency')
#         plt.yscale('log')
#         # plt.tight_layout()
#         plt.show()
        
#         # plt.savefig(f"original_data_plot/batch_{batch}/layer_{layer}.png")
#         # plt.close()
