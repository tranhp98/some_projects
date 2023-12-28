import torch
import copy
import wandb
from tqdm import tqdm
from utils import dict_append
from properties_checker import compute_linear_approx, compute_smoothness

# =====================================
# Standard Training (Benchmark)
# =====================================

def train_step(epoch, net, trainloader, criterion, optimizer, device):
    '''
    Train single epoch.
    '''
    print(f'\nTraining epoch {epoch+1}..')
    net.train()
    convexity_gap = 0
    L = 0
    num = 0
    denom = 0
    prev_loss = 0
    current_loss = 0
    train_loss = 0
    exp_avg_L_1 = 0
    exp_avg_L_2 = 0
    exp_avg_gap_1 = 0
    exp_avg_gap_2 = 0
    step = 0
    train_acc = 0
    total = 0
    correct = 0
    prev_grad = [torch.zeros_like(p) for p in net.parameters()]
    prev_param = [torch.zeros_like(p) for p in net.parameters()] 
    current_grad = [torch.zeros_like(p) for p in net.parameters()]
    current_param = [torch.zeros_like(p) for p in net.parameters()] 
    # pbar = tqdm(enumerate(trainloader))
    iterator = enumerate(trainloader)
    prev_batch = next(iterator)
    # final_loss = 0
    # print(pbar)
    for batch, (inputs, labels) in iterator:
        # load data
        # print("current iteration", batch)
        inputs, labels = inputs.to(device), labels.to(device)
        # if step >0: #updating prev_prev_param
        #     prev_loss = current_loss
        #     optimizer.save_prev_param()
        
        #compute \nabla f(w_t,x_{t-1})
        prev_batch_image = prev_batch[1][0].to(device)
        prev_batch_target = prev_batch[1][1].to(device)
        prev_batch_outputs = net(prev_batch_image) 
        prev_batch_loss = criterion(prev_batch_outputs, prev_batch_target) #f(w_t,x_{t-1})
        current_loss = prev_batch_loss.item() 
        prev_batch_loss.backward()
        i = 0
        with torch.no_grad():
            for p in net.parameters():
                current_grad[i].copy_(p.grad) #\nabla f(w_t,x_{t-1})
                current_param[i].copy_(p) #w_t
                i+=1
        # zero grad to do the actual update
        optimizer.zero_grad()

        # forward and backward propagation
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # if i ==0:
        #     prev_loss = loss.item()
        loss.backward()
        # if epoch ==0 and i == 0:
        #     optimizer.save_param()
        #     prev_loss = loss.item()
        # final_loss = loss.item()
        if step >0:
            # prev_loss = current_loss
            # # print("prev s", current_loss)
            # # print("current loss", loss.detach().float() )
            # i = 0
            # with torch.no_grad():
            #     for p in model.parameters():
            #         prev_grad[i].copy_(current_grad[i])
            #         prev_param[i].copy_(current_param[i])
            #         i+=1
            # get the inner product
            linear_approx = compute_linear_approx(current_param, current_grad, prev_param)
            # get the smoothness constant, small L means function is relatively smooth
            current_L = compute_smoothness(current_param, current_grad, prev_param, prev_grad)
            L = max(L,current_L)
            # L = max(L,compute_smoothness(model, current_param, current_grad))
            # this is another quantity that we want to check: linear_approx / loss_gap. The ratio is positive is good
            num+= linear_approx
            denom+= current_loss - prev_loss # f(w_t,x_{t-1}) - f(w_{t-1},x_{t-1})
            current_convexity_gap = current_loss - prev_loss - linear_approx 
            exp_avg_gap_1 = 0.99*exp_avg_gap_1 + (1-0.99)*current_convexity_gap
            exp_avg_gap_2 = 0.9999*exp_avg_gap_2 + (1-0.9999)*current_convexity_gap
            exp_avg_L_1 = 0.99*exp_avg_L_1+ (1-0.99)*current_L
            exp_avg_L_2 = 0.9999*exp_avg_L_2+ (1-0.9999)*current_L
            convexity_gap+= current_convexity_gap

        i = 0
        with torch.no_grad():
            for p in net.parameters():
                prev_grad[i].copy_(p.grad) #hold \nabla f(w_{t-1},x_{t-1}) for next iteration
                prev_param[i].copy_(p) # hold w_{t-1 } for next iteration
                i+=1
        optimizer.step()
        prev_loss = loss.item() 
        prev_batch = (batch, (inputs, labels))
        # current_loss = loss.item()
        step+=1
        # stat updates
        # inner_product += inner
        train_loss += (loss.item() - train_loss)/(batch+1)  # average train loss
        total += labels.size(0)                             # total predictions
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()        # correct predictions
        train_acc = 100*correct/total                       # average train acc
        optimizer.zero_grad()
        # prev_loss = loss.item()
        # pbar.set_description(f'epoch {epoch+1} batch {batch+1}: \
        #     train loss {train_loss:.2f}, train acc: {train_acc:.2f}, smoothness_constant:{L:.2f}, convexity_gap:{convexity_gap:.2f} ' )
        # # wandb.log({ 'smoothness_constant': L,'convexity_gap': convexity_gap })
    prev_batch_image = prev_batch[1][0].to(device)
    prev_batch_target = prev_batch[1][1].to(device)
    prev_batch_outputs = net(prev_batch_image) 
    prev_batch_loss = criterion(prev_batch_outputs, prev_batch_target) #f(w_t,x_{t-1})
    current_loss = prev_batch_loss.item() 
    prev_batch_loss.backward()
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            current_grad[i].copy_(p.grad) #\nabla f(w_t,x_{t-1})
            current_param[i].copy_(p) #w_t
            i+=1
    # zero grad to do the actual update
    optimizer.zero_grad()
    return prev_grad, prev_param, current_grad, current_param, prev_loss, current_loss, convexity_gap/step, L,num/denom, num, denom,exp_avg_L_1, exp_avg_L_2, exp_avg_gap_1, exp_avg_gap_2, train_acc


def test_step(epoch, net, testloader, criterion, device):
    '''
    Test single epoch.
    '''
    print(f'\nEvaluating epoch {epoch+1}..')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(testloader))
    with torch.no_grad():
        for batch, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += (loss.item() - test_loss)/(batch+1)
            total += labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            # test_acc = 100*correct/total
            # pbar.set_description(f'test loss: {test_loss}, test acc: {test_acc}')

    test_acc = 100*correct/total
    print(f'test loss: {test_loss}, test acc: {test_acc}')
    return test_loss, test_acc


def train(net, trainloader, testloader, epochs, criterion, optimizer, scheduler, device,args):
    stats = {
        'args': None,
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    filename = "cnn_checkpoint/0.pth.tar"
    torch.save({'state_dict':optimizer.state_dict(), 'model_dict': net.state_dict() }, filename)
    for epoch in range(epochs):
        # name = 'checkpoint/'+ str(epoch+1) + ".pth.tar"
        # saved_checkpoint = torch.load(name)
        # optimizer.save_param(saved_checkpoint['state_dict'])
        # prev_loss = saved_checkpoint['current_loss']
        print("current epoch", epoch)
        prev_grad, prev_param, current_grad, current_param, prev_loss, current_loss, convexity_gap, smoothness,ratio,num, denom,exp_avg_L_1, exp_avg_L_2, exp_avg_gap_1, exp_avg_gap_2, train_acc = train_step(epoch, net, trainloader, criterion, optimizer, device)
        # inner_product = inner
        if args.save:
            filename = "cnn_checkpoint/" + str(epoch+1) + ".pth.tar"
            torch.save({'state_dict':optimizer.state_dict(),'prev_grad':prev_grad, 
                            'prev_param': prev_param, 'current_grad':current_grad, 'current_param': current_param
                            , 'prev_loss':prev_loss , 'current_loss': current_loss
                          , 'model_dict': net.state_dict() }, filename)
            # save_param[epoch] = {'state_dict': copy.deepcopy(optimizer.state_dict()), 'loss':loss}
        test_loss, test_acc = test_step(epoch, net, testloader, criterion, device)

        dict_append(stats,
            test_loss=test_loss, test_acc=test_acc)
        wandb.log(
        {
            "train_loss": current_loss,
            "convexity_gap": convexity_gap,
            "smoothness": smoothness,
            "linear/loss_gap": ratio,
            "numerator" : num,
            "denominator": denom,
            'exp_avg_L_.99': exp_avg_L_1,
            'exp_avg_L_.9999': exp_avg_L_2, 
            "exp_avg_gap_.99":  exp_avg_gap_1, 
            "exp_avg_gap_.9999":  exp_avg_gap_2, 
            "train_accuracy": train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'learning rate': scheduler.get_lr()[0]
        }
    )
        # wandb.log({
        #     'test_loss': test_loss, 'test_acc': test_acc, 'learning rate': scheduler.get_lr()[0]})
        scheduler.step()
    return stats



# =====================================
# Private Training (DP-OTB)
# =====================================
'''
Train with DP-OTB.

To implement DP-OTB, we need to networks: one original network (net) corresponding to the OCO algorithm and
one clone network (net_clone) corresponding to DP-OTB.
*At the end, we copy net <- net_clone because we want to output the aggregated weights.

We stick to the notation in the paper, with k = 1 (i.e., beta_t = t).
'''

# def beta(t):
#     '''
#     \beta_t = t^k. Here we choose k = 1.
#     '''
#     return t


