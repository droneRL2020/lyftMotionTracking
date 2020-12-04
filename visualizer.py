    # def visualize_data(self, data, loss): 
    #     if (loss.item() > 100):
    #         print("[loss] :",  loss.item())
    #     for i in range(32):
    #         image = data["image"][i]
    #         image = image.cpu().numpy()
    #         image = image.transpose(1, 2, 0)
    #         image = self.train_dataset.rasterizer.to_rgb(image)
    #         image = image.transpose(2, 0, 1)
    #         plt.imshow(np.uint8(image))
    #         plt.show()