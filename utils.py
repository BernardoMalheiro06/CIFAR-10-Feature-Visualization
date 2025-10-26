import torch


def optimize_hidden_layer_latent(net, decoder, layers, num_features, latent_size, layer_name, device, num_iterations=1000, apply_laplace_kernel=False, convolutional=True):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    g = torch.Generator()
    g.manual_seed(42)

    net.eval()
    decoder.eval()

    latent_vector = torch.randn(num_features, latent_size, device = device, requires_grad=True)
    latent_vector = latent_vector.to(device)
    latent_optimizer = torch.optim.Adam([latent_vector], lr=0.01)

    for _ in range(num_iterations):
        latent_optimizer.zero_grad()
        net.zero_grad()
        decoder.zero_grad()

        img = decoder(latent_vector)
        img = img.to(device)

        net(img)

        activations = layers[layer_name].to(device)

        losses = []
        for j in range(num_features):
            if convolutional:
                loss_j = torch.exp(-activations[j, j, :, :]).mean()
            else:
                loss_j = torch.exp(-activations[j, j]).mean()
            losses.append(loss_j)

        loss = torch.stack(losses).mean()

        if apply_laplace_kernel:
            laplace_kernel = torch.tensor(
                [[0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]],
                dtype=torch.float32, device=device
            ).unsqueeze(0).unsqueeze(0).to(device)
            laplace_out = torch.nn.functional.conv2d(img, laplace_kernel.repeat(img.size(1), 1, 1, 1), padding = 1, groups = img.size(1),)
            laplace_kernel_loss = laplace_out.mean() * 0.000001
            loss += laplace_kernel_loss
        
        loss.backward()
        latent_optimizer.step()

    return latent_vector

def optimize_hidden_layer_decorr_latent(net, decoder, layers, num_features, latent_size, layer_name, device, W_inv, miu, num_iterations=1000, apply_laplace_kernel=False, convolutional=True):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    g = torch.Generator()
    g.manual_seed(42)

    net.eval()
    decoder.eval()

    latent_vector = torch.randn(num_features, latent_size, device = device, requires_grad=True)
    latent_vector = latent_vector.to(device)
    latent_optimizer = torch.optim.Adam([latent_vector], lr=0.01)

    for _ in range(num_iterations):
        latent_optimizer.zero_grad()
        net.zero_grad()
        decoder.zero_grad()

        img = decoder(latent_vector @ W_inv + miu)
        img = img.to(device)

        net(img)

        activations = layers[layer_name].to(device)

        losses = []
        for j in range(num_features):
            if convolutional:
                loss_j = torch.exp(-activations[j, j, :, :]).mean()
            else:
                loss_j = torch.exp(-activations[j, j]).mean()
            losses.append(loss_j)

        loss = torch.stack(losses).mean()

        if apply_laplace_kernel:
            laplace_kernel = torch.tensor(
                [[0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]],
                dtype=torch.float32, device=device
            ).unsqueeze(0).unsqueeze(0).to(device)
            laplace_out = torch.nn.functional.conv2d(img, laplace_kernel.repeat(img.size(1), 1, 1, 1), padding = 1, groups = img.size(1),)
            laplace_kernel_loss = laplace_out.mean() * 0.000001
            loss += laplace_kernel_loss
        
        loss.backward()
        latent_optimizer.step()

    return latent_vector

def optimize_output_layer_latent(net, decoder, num_features, latent_size, device, num_iterations=1000, apply_laplace_kernel=True):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    g = torch.Generator()
    g.manual_seed(42)

    net.eval()
    decoder.eval()

    latent_vector = torch.randn(num_features, latent_size, device = device, requires_grad=True)
    latent_vector = latent_vector.to(device)
    latent_optimizer = torch.optim.Adam([latent_vector], lr=0.01)

    for _ in range(num_iterations):
        latent_optimizer.zero_grad()
        net.zero_grad()
        decoder.zero_grad()

        img = decoder(latent_vector)
        img = img.to(device)

        output = net(img)

        losses = []
        for j in range(num_features):
            loss_j = torch.exp(-output[j, j]).mean()
            losses.append(loss_j)

        loss = torch.stack(losses).mean()

        if apply_laplace_kernel:
            laplace_kernel = torch.tensor(
                [[0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]],
                dtype=torch.float32, device=device
            ).unsqueeze(0).unsqueeze(0).to(device)
            laplace_out = torch.nn.functional.conv2d(img, laplace_kernel.repeat(img.size(1), 1, 1, 1), padding = 1, groups = img.size(1),)
            laplace_kernel_loss = laplace_out.mean() * 0.000001
            loss += laplace_kernel_loss
        
        loss.backward()
        latent_optimizer.step()
    

    return latent_vector

def optimize_output_layer_decorr_latent(net, decoder, num_features, latent_size, device, W_inv, miu, num_iterations=1000, apply_laplace_kernel=True):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    g = torch.Generator()
    g.manual_seed(42)

    net.eval()
    decoder.eval()

    latent_vector = torch.randn(num_features, latent_size, device = device, requires_grad=True)
    latent_vector = latent_vector.to(device)
    latent_optimizer = torch.optim.Adam([latent_vector], lr=0.01)

    for _ in range(num_iterations):
        latent_optimizer.zero_grad()
        net.zero_grad()
        decoder.zero_grad()

        img = decoder(latent_vector @ W_inv + miu)
        img = img.to(device)

        output = net(img)

        losses = []
        for j in range(num_features):
            loss_j = torch.exp(-output[j, j]).mean()
            losses.append(loss_j)

        loss = torch.stack(losses).mean()

        if apply_laplace_kernel:
            laplace_kernel = torch.tensor(
                [[0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]],
                dtype=torch.float32, device=device
            ).unsqueeze(0).unsqueeze(0).to(device)
            laplace_out = torch.nn.functional.conv2d(img, laplace_kernel.repeat(img.size(1), 1, 1, 1), padding = 1, groups = img.size(1),)
            laplace_kernel_loss = laplace_out.mean() * 0.000001
            loss += laplace_kernel_loss
        
        loss.backward()
        latent_optimizer.step()
    

    return latent_vector