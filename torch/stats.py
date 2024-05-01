import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train_mlp_battery_1.csv')
df.head()

plt.figure(figsize=(4,2))
plt.plot(df['Epoch'], df['train_Loss'], label='train_loss')
plt.plot(df['Epoch'], df['val_Loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('figures/loss_battery_1.png')




COLOR_Ro = "#69b3a2"
COLOR_qMax = "#3399e6"

fig, ax1 = plt.subplots(figsize=(4,2))
# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  
ax1.plot(df['Epoch'], df['Ro'] * 1.0e1, color=COLOR_Ro)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Ro (Ohm)', color=COLOR_Ro)


ax2.plot(df['Epoch'], df['qMax'] * 1.0e4, color=COLOR_qMax)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('qMax (Coloumbs)', color=COLOR_qMax)
ax2.yaxis.label.set_color(COLOR_qMax)
ax2.tick_params(axis='y', colors=COLOR_qMax)
plt.tight_layout()

# plt.show()
plt.savefig('figures/Ro_qMax_1.png')
