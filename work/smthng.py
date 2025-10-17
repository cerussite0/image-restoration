###############========Language Model Head TRAINING LOOP===========################

if __name__ == "__main__":

    transform = PairedTransform(size=(256, 256))
    image_loader = MultiTaskLoader(root_dir='/kaggle/input/', batch_size=batch_size,transform=transform)
    prompt_loader = MultiTaskPromptLoader('/kaggle/input/prompt-dataset/D5_task_prompt.json', batch_size=batch_size)
    ######################====== LANGUAGE MODEL INITIALIZATION ======#############################
    LMODEL = 'TaylorAI/bge-micro-v2'
    language_model = LanguageModel(model=LMODEL).to(device).eval() # Keep frozen
    lm_head = LMHead(embedding_dim=384, hidden_dim=256, num_classes=5).to(device)
    # Visualization config
    VIS_DIR = os.path.join(checkpoint_dir, "visualizations")
    os.makedirs(VIS_DIR, exist_ok=True)
    ######################====== IMAGE MODEL INITIALIZATION ======#############################
    image_model = create_model(input_channels = 3, width = 32, enc_blks = [2, 2, 4, 8], middle_blk_num = 4, dec_blks = [2, 2, 2, 2], txtdim=256)
    image_model = image_model.to(device)
    
    ##############=======Parameter Count===========##########   
    def count_trainable_parameters(model):
        """
        Counts the number of trainable parameters in a PyTorch model.
    
        Args:
            model: PyTorch model
        
        Returns:
            total_params: Total number of trainable parameters
        """
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params
    ################======== Count trainable parameters===========################
    trainable_params = count_trainable_parameters(image_model)
    print(f"Total trainable parameters: {trainable_params:,}")
    ######################====== Loss and Optimizer ======#############################
    criterion_class = nn.CrossEntropyLoss()
    
    # LM Head-specific optimizer
    optimizer_lm = torch.optim.AdamW(lm_head.parameters(), lr=5e-4)

    print("Starting Stage 1: LM Head Training")
    for epoch in range(1,num_epochs_lm+1):
        
        lm_head.train()
        for _ in range(1,2000,batch_size):
        
            # Random task selection
            task = random.choice([ 'haze', 'noise','rain', 'blur', 'lol'])
        
            # Get batch
            texts,labels = prompt_loader.get_batch(task)
            labels = labels.to(device)

            # Forward pass through frozen language model
            with torch.no_grad():
                embeddings = language_model(texts)

             
            # LM head forward
            _, logits = lm_head(embeddings)
            lm_loss = criterion_class(logits, labels)
            
            # Backpropagate
            optimizer_lm.zero_grad()
            lm_loss.backward()
            optimizer_lm.step()            
    
        train_acc = calculate_accuracy(prompt_loader, language_model, lm_head, device) 
        print(f"Epoch [{epoch}/{num_epochs_lm}] | " f"Train Acc: {train_acc:.2f}%")

    #################MAIN TRAINING LOOP#################


    criterion_class = nn.CrossEntropyLoss()
    criterion_image = nn.L1Loss()


    optimizer_lm = optim.Adam(lm_head.parameters(), lr=1e-3)
    
    # Unified AdamW optimizer with learning rate 5e-4
    optimizer = torch.optim.AdamW([
        {'params': lm_head.parameters()},
        {'params': image_model.parameters()}
        ], lr=5e-4)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


    print("Starting Stage 2: Main Training")
    # Main training loop
    for epoch in range(1,num_epochs+1):
        lm_head.train()
        image_model.train()
    
        for _ in range(0,2000,batch_size):

            # Random task selection
            task = random.choice(['haze', 'noise', 'rain', 'blur', 'lol'])
        
            # --- Language Model Head Training ---
            texts,labels = prompt_loader.get_batch(task)
            inputs, targets = image_loader.get_batch(task)

            # Move data to device
            labels = labels.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # --- Forward passes ---
            # Text embeddings (frozen language model)
            with torch.no_grad():
                embeddings = language_model(texts)
            
            # LM head forward
            text_embd, logits = lm_head(embeddings)

            # Image model forward
            outputs = image_model(inputs, text_embd)
            
            # --- Loss calculation ---
            loss_image = criterion_image(outputs, targets)
            loss_class = criterion_class(logits, labels)
            total_loss = loss_image + 0.3 * loss_class  # Combined loss

            # --- Backward pass and optimize ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Update learning rate
        scheduler.step()    
        
        # Epoch evaluation
        print(f"\nEpoch {epoch} Evaluation:")
    
        # 1. LM Head Accuracy
        train_acc = calculate_accuracy(prompt_loader, language_model, lm_head, device)
        print(f"Classification Accuracy: {train_acc:.2f}%")

        # 2. PSNR Table
        psnr_values = evaluate_model(image_model, lm_head, image_loader, device)
    
        if epoch % 50 == 0:
            # Save models
            checkpoint_path = os.path.join(checkpoint_dir, f"joint_epoch_{epoch}.pth")
            torch.save({
                'image_model': image_model.state_dict(),
                'lm_head': lm_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, checkpoint_path)
        
            # Save visuals
            save_visuals(image_model, lm_head, image_loader, epoch, device)
            print(f"Saved checkpoint and visuals for epoch {epoch}")

        if epoch%100==0:
            break