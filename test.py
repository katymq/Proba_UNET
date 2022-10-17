import matplotlib.pyplot as plt
import torch
import numpy as np

def test(net, loss_fn, dataloaders_test, device, ProbaUnet = True, plot_ = True):
    loss_classes =  np.array([[0,0,0]])
    loss = []
    for _, batch in enumerate(dataloaders_test): 
        X, y = batch

        X, y = X.to(device), y.to(device)
        
        if ProbaUnet:
            _,_, pred =  net(X, y)
        else:
            pred =  net(X)

        seg = torch.argmax(pred, dim=1).numpy()[0]# torch.argmax(pred, 1).numpy()  # Get  prediction classes
        loss_v = loss_fn(pred, y)
        loss_classes_  = []
        for i in range(pred.shape[1]):
            y1 = 1*(y==i)
            pred1 = pred[:,i,:,:].unsqueeze(0)
            loss_classes_.append(-(loss_fn(pred1, y1).item()-1))
        #loss_classes_ , loss_v = loss_fn(pred, y, Test = True)
        loss_classes  = np.concatenate((loss_classes, np.array([loss_classes_])), axis = 0)
        loss.append(loss_v.item())
        print(loss_v)
        if plot_:
            print(loss_classes_)
            _, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(X.squeeze().permute(1,2,0))
            ax2.imshow(seg)
            ax3.imshow(y.squeeze())
            plt.show()
    return loss_classes[1:,:].mean(0), np.mean(loss)

def test_SR(net, loss_fn, dataloaders_test, device, ProbaUnet = True, plot_ = True):
    for _, batch in enumerate(dataloaders_test): 
        X, y = batch

        X, y = X.to(device), y.to(device)
        
        if ProbaUnet:
            _,_, pred =  net(X, y)
        else:
            pred =  net(X)

        seg = torch.argmax(pred, dim=1).numpy()[0]# torch.argmax(pred, 1).numpy()  # Get  prediction classes
        for i in range(pred.shape[1]):
            y1 = 1*(y==i)
            pred1 = pred[:,i,:,:].unsqueeze(0)
        if plot_:
            _, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(X.squeeze().permute(1,2,0))
            ax2.imshow(seg)
            ax3.imshow(y.squeeze())
            plt.show()
