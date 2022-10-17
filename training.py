import os
import torch 
import numpy as np
import matplotlib.pyplot as plt

def training_probaUnet(dataloaders, epochs, device, beta, net, optimizer, modelPath):
    train_loss = [] 
    for epoch in range(epochs):
        for step, batch in enumerate(dataloaders): 
            X, y = batch
            patch, mask = X.to(device), y.to(device)
            
            rec_loss,  dkl_loss, _ =  net(patch,mask)      
            elbo =  -(rec_loss + beta * dkl_loss)
            if step % 30 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Recons: {}  dkl : {}'.format(step, -elbo, rec_loss, dkl_loss))
                    
            loss = -elbo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = loss / len(dataloaders.dataset)
        train_loss.append(epoch_loss)
        if epoch % 5 == 0: #Save model weight once every 60k steps permenant file
                print("Saving Model" +str(epoch) + ".torch")
                torch.save(net.state_dict(),   os.path.join(modelPath,str(epoch) + ".torch") )

    train_loss_ = [train_loss[i].item() for i in range(len(train_loss))]
    plt.plot(train_loss_)
    plt.show()

    np.save(os.path.join(modelPath,'loss'+str(epoch) + ".npy"), train_loss_)

    return train_loss, train_loss_

def training_Unet(dataloaders, epochs, device, loss_fn, net, optimizer, modelPath):
    train_loss = [] 
    for epoch in range(epochs):
        for step, batch in enumerate(dataloaders): 
            X, y = batch
            X, y  = X.to(device), y.to(device)
            outputs = net(X)
            loss = loss_fn(outputs, y)

            if step % 30 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  '.format(step, loss))
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = loss / len(dataloaders.dataset)
        train_loss.append(epoch_loss)
        if epoch % 5 == 0: #Save model weight once every 60k steps permenant file
                print("Saving Model" +str(epoch) + ".torch")
                torch.save(net.state_dict(),   os.path.join(modelPath,str(epoch) + ".torch") )
    
    train_loss_ = [train_loss[i].item() for i in range(len(train_loss))]
    plt.plot(train_loss_)
    plt.show()
    np.save(os.path.join(modelPath,'loss'+str(epoch) + ".npy"), train_loss_)
    return train_loss, train_loss_