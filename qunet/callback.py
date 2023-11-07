class Callback:
    def __init__(self):
        """
        Abstract base class used to build new callbacks.
        Subclass this class and override any of the relevant hooks.

        Example:
        ------------                
        ```
        class MyCallback(Callback):
	        def on_train_epoch_start(self, trainer, pl_module):
		        if trainer.epoch % 10 == 0:
			        trainer.data.val = new_data

        trainer = Trainer(model, data_trn, data_val, callbacks=[MyCallback()])
        trainer.fit()
        ```
        

        See also:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.htm
        """

    def on_fit_start(self, trainer, model):
        """ 
        Called when fit begins. 
        """
        pass
    #---------------------------------------------------------------------------

    def on_fit_end(self, trainer, model):
        """ 
        Called when fit ends.
        """
        pass

    #---------------------------------------------------------------------------

    def on_predict_start(self, trainer, model):
        """
        Called when the predict begins.
        """
        pass
    #---------------------------------------------------------------------------

    def on_predict_end(self, trainer, model):
        """
        Called when predict ends.
        """
        pass

    #---------------------------------------------------------------------------

    def on_epoch_start(self, trainer, model):
        """
        Called when epoch in fit begins.
        """
        pass
    #---------------------------------------------------------------------------

    def on_epoch_end(self, trainer, model):
        """
        Called when epoch in fit ends.
        """
        pass

    #---------------------------------------------------------------------------

    def on_train_epoch_start(self, trainer, model):
        """
        Called when the train begins.
        """
        pass

    #---------------------------------------------------------------------------

    def on_train_epoch_end(self, trainer, model):
        """
        Called when the train ends.
        """
        pass

    #---------------------------------------------------------------------------

    def on_validation_epoch_start(self, trainer, model):
        """
        Called when the validation loop begins.
        """
        pass

    #---------------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer, model):
        """
        Called when the validation loop ends.
        """
        pass

    #---------------------------------------------------------------------------  
    #                                 Batches
    #---------------------------------------------------------------------------  
        
    def on_train_before_batch_transfer(self, trainer, model, batch, batch_id):
        """
        Called before transfer batch to GPU
        """
        return batch

    #---------------------------------------------------------------------------
    #     
    def on_train_after_batch_transfer(self, trainer, model, batch, batch_id):
        """
        Called after transfer batch to GPU
        """
        return batch

    #---------------------------------------------------------------------------
    #
    def on_train_batch_end(self, trainer, model, batch, batch_id):
        """
        Called after train batch
        """
        return

    #---------------------------------------------------------------------------  
        
    def on_validation_before_batch_transfer(self, trainer, model, batch, batch_id):
        """
        Called before transfer batch to GPU
        """
        return batch

    #---------------------------------------------------------------------------
    #     
    def on_validation_after_batch_transfer(self, trainer, model, batch, batch_id):
        """
        Called after transfer batch to GPU
        """
        return batch

    #---------------------------------------------------------------------------
    #
    def on_validation_batch_end(self, trainer, model, batch, batch_id):
        """
        Called after train batch
        """
        return

    #---------------------------------------------------------------------------
    #     
    def on_predict_before_batch_transfer(self, trainer, model, batch, batch_id):
        """
        Called before transfer batch to GPU
        """
        return batch

    #---------------------------------------------------------------------------
    #     
    def on_predict_after_batch_transfer(self, trainer, model, batch, batch_id):
        """
        Called after transfer batch to GPU
        """
        return batch
    

    #---------------------------------------------------------------------------

    def on_after_step(self, trainer, model, batch, batch_id):
        """
        Called after optimizer.step
        """
        pass



    #---------------------------------------------------------------------------
    #                                   Saves
    #---------------------------------------------------------------------------

    def on_best_score(self, trainer, model):
        """
        Called when get new best score.
        """
        pass

    #---------------------------------------------------------------------------

    def on_best_loss(self, trainer, model):
        """
        Called when get new best loss.
        """
        pass

    #---------------------------------------------------------------------------

    def on_save_checkpoint(self, trainer, model, checkpoint):
        """
        Called when saving a checkpoint to give you a chance to store anything else you might want to save.

        Args:
        ------------
        trainer (Trainer):
            the current Trainer instance.
        modele (nn.Module): 
            the current LightningModule instance.
        checkpoint (Dict[str, Any]):
            the checkpoint dictionary that will be saved.        
        """
        pass

    #---------------------------------------------------------------------------

    def on_after_plot(self, trainer, model):
        """
        Called after plot_period.
        """
        pass


