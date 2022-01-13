# Torte
### Versatile Trainer

- Minimal
```Python
from torte import Trainer as TorteTrainer

class Trainer(TorteTrainer):
    def __init__(self, model, dataset, batch_size=512):
        self.loss_fn = torch.nn.NLLLoss()
        optimizer = Adam(model.parameters(), lr=0.0001)
        super().__init__(model, [optimizer], dataset, batch_size=batch_size)

    def single_epoch(self, **kwargs):
        avg_loss = 0
        for batch in self.data_loader:
            self.optimizers[0].zero_grad()
            x, y = batch
            x_obs, x_pi = x
            if is_cuda():
                y = y.cuda(self.device_id)
                x_obs, x_pi = x_obs.cuda(self.device_id), x_pi.cuda(self.device_id)
            log_prob = self.model(x_obs, x_pi).squeeze(1)
            loss = self.loss_fn(log_prob, y)
            loss.backward()
            avg_loss += loss.item()
            self.optimizers[0].step()
        print(f'Epoch loss={avg_loss / len(self.data_loader)}')
        return avg_loss
```
- Automatic tracking parameters

```s```