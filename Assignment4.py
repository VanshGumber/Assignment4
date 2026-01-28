import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data.csv", encoding="latin1", low_memory=False)

x = data["no2"].dropna().values.astype(np.float32)

x = (x - x.mean()) / x.std()

r = 102303922

a_r = 0.5 * (r % 7)
b_r = 0.3 * (r % 5 + 1)

z = x + a_r * np.sin(b_r * x)

print("a_r =", a_r)
print("b_r =", b_r)

z_tensor = torch.tensor(z, dtype=torch.float32).view(-1, 1)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=1e-3)
opt_D = torch.optim.Adam(D.parameters(), lr=1e-3)

epochs = 8000
batch_size = 128

for epoch in range(epochs):
    idx = torch.randint(0, len(z_tensor), (batch_size,))
    real_samples = z_tensor[idx]

    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    noise = torch.randn(batch_size, 1)
    fake_samples = G(noise)

    d_loss = (
        criterion(D(real_samples), real_labels) +
        criterion(D(fake_samples.detach()), fake_labels)
    )

    opt_D.zero_grad()
    d_loss.backward()
    opt_D.step()

    g_loss = criterion(D(fake_samples), real_labels)

    opt_G.zero_grad()
    g_loss.backward()
    opt_G.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")


with torch.no_grad():
    z_fake = G(torch.randn(50000, 1)).numpy().flatten()

plt.figure(figsize=(8, 5))
sns.kdeplot(z, label="Real z", linewidth=2)
sns.kdeplot(z_fake, label="GAN Generated z", linewidth=2)
plt.xlabel("z")
plt.ylabel("Density")
plt.title("PDF of Transformed Variable z using GAN")
plt.legend()
plt.grid(True)
plt.show()
