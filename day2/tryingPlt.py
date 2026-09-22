import numpy as np
import matplotlib.pyplot as plt

def main():
    time = np.linspace(0,10,500)
    amplitude = np.sin(2*np.pi*time) * np.exp(-0.5 * np.pi * time)

    fig, ax = plt.subplots(figsize=(8,4.5), dpi=100)
    ax.plot(time,amplitude,color='crimson',linewidth=2,label="sine wave")

    ax.set_title("trying plt",fontsize=14,fontweight='bold',pad=15)
    ax.set_xlabel('Time',fontsize=12)
    ax.set_ylabel('displacement',fontsize=12)
    ax.grid(True,linestyle='--',alpha=0.6)
    ax.legend(loc='upper right',frameon=True,shadow=True)
    plt.show()
if __name__ == "__main__":
    main()
