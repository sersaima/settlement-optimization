# Importing libraries
import matplotlib.pyplot as plt
import numpy as np


def avg_metric(batch_metrics, metric_name):
    return np.mean([bm[metric_name] for bm in batch_metrics])


def plot_before_after(batch_metrics_1, batch_metrics_2):
    l1 = len(batch_metrics_1)
    l2 = len(batch_metrics_2)

    X = np.arange(0, l1+l2, 1)

    ts = np.concatenate(([100*bm["settle_rate"] for bm in batch_metrics_1],
                         [100*bm["settle_rate"] for bm in batch_metrics_2]))

    cu = np.concatenate(([bm["total_loan"] for bm in batch_metrics_1],
                         [bm["total_loan"] for bm in batch_metrics_2]))


    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)
    ax2 = ax.twinx()
    ax.plot(cu, '-o', label="Net collateral used")
    ax2.plot(ts, '-o', color="C2", label="% of Transactions settled")

    ax.set_title('Batch transaction settlement optimization')

    ax.legend(loc=2)
    ax2.legend(loc=1)
    ax2.set_ylim([20, 100])
    ax.set_ylim([0.75*cu.min(), 1.1*cu.max()])

    plt.axvline(x=l1-0.5, color='r', ls='--')

    plt.text(l1-10, 70, 'New regulation:\n\n          loans are bad', fontsize = 14, color='r')


    ax.set_xlabel("Batch №")
    ax.set_ylabel("$")
    ax2.set_ylabel("%")
    plt.show()



def plot_single_run(batch_metrics):
    l = len(batch_metrics)
    X = np.arange(0, l, 1)

    ts = np.array([100*bm["settle_rate"] for bm in batch_metrics])

    cu = np.array([bm["total_loan"] for bm in batch_metrics])

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)
    ax2 = ax.twinx()
    ax.plot(cu, '-o', label="Net collateral used")
    ax2.plot(ts, '-o', color="C2", label="% of Transactions settled")

    ax.set_title('Batch transaction settlement optimization')

    str1 = f'Avg Settlement Success:  {"{0:0.2f}".format(100*avg_metric(batch_metrics, "settle_rate"))}%\n'
    str2 = f'Avg Auto-Collateral Used:  ${"{0:0.0f}".format(avg_metric(batch_metrics, "total_loan"))}'
        
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    

    ax.legend(loc=2)
    ax2.legend(loc=1)
    ax2.set_ylim([20, 100])
    ax.set_ylim([0.75*cu.min(), 1.1*cu.max()])

    ax.set_xlabel("Batch №")
    ax.set_ylabel("$")
    ax2.set_ylabel("%")

    ax2.text(0.3, 0.95, str1 + str2, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.show()


# def plot2_before_after(batch_metrics_before, batch_metrics_after_same, batch_metrics_after_adapted):
#     X_old = np.arange(0, len(batch_metrics_before.keys()), 1)
#     X_new = np.arange(len(batch_metrics_before.keys()), len(batch_metrics_before.keys()) + len(batch_metrics_after_same.keys()), 1)

#     settled_old1 = np.array([100*bm["settle_rate"] for bm in batch_metrics_before])

#     settled_old2 = np.array([100*bm["settle_rate"] for bm in batch_metrics_after_same])
#     settled_old2 = np.concatenate(([settled_old1[-1]], settled_old2))

#     settled_new2 = np.array([100*bm["settle_rate"] for bm in batch_metrics_after_adapted])
#     settled_new2 = np.concatenate(([settled_old1[-1]], settled_new2))


#     penalty_old1 = np.random.normal(500, 2.25, 30)
#     penalty_old1 = np.multiply(penalty_old1, (100 - settled_old1) / 100)
#     penalty_old2 = np.random.normal(500, 2.25, 20)
#     penalty_old2 = np.concatenate(([1], penalty_old2))
#     penalty_old2 = np.multiply(penalty_old2, (100 - settled_old2) / 100)
#     penalty_old2[0] = sum(penalty_old1)

#     penalty_new2 = np.random.normal(500, 2.25, 20)
#     penalty_new2 = np.concatenate(([1], penalty_new2))
#     penalty_new2 = np.multiply(penalty_new2, (100 - settled_new2) / 100)
#     penalty_new2[0] = sum(penalty_old1)


#     print(sum(penalty_old1))
#     print(np.cumsum(penalty_old1))
#     print(penalty_old2)
#     fig, axs = plt.subplots(2, 1)
#     fig.set_size_inches(16, 10)

#     axs[0].plot(X_old, settled_old1, '-o', color='green')
#     axs[0].plot(X_new, settled_old2, '--o', color='green')
#     axs[0].plot(X_new, settled_new2, '-o', color='orange')
#     axs[0].set_ylim([80, 100])



#     axs[1].bar(X_new, np.cumsum(penalty_old2), color='green')
#     axs[1].bar(X_new, np.cumsum(penalty_new2), color='orange', label='Adapted optimizer')
#     axs[1].bar(X_old, np.cumsum(penalty_old1), color='green', label='Old optimizer')
#     # axs[1].set_ylim([80, 100])

#     axs[1].legend()


#     axs[0].axvline(x=29.5, color='r', ls='--')


#     axs[0].text(21, 82, 'New regulation:\n\n     large penalties for\n     failed trades', fontsize = 12, color='r')
#     # plt.text(26, 95, 'New regulation:\n\n     large penalties for\n     failed trades', fontsize = 14, color='r')

#     axs[0].set_title("% of settled transactions")
#     axs[1].set_title("and corresponding cumulative penalties for failed")
#     axs[0].set_ylabel("%")
#     axs[1].set_xlabel("Batch №")
#     axs[1].set_ylabel("$")
#     plt.show()