from Model import solve_ntsp
from DataModels import *
from DataInput import generate_ntsp_input, ntsp_input_from_batch
from DataVisual import plot_before_after, plot_single_run, avg_metric
import pandas as pd


if __name__ == "__main__":

    # --- SINGLE BATCH ---

    batch = pd.read_csv("settlement_optimization/Data/transactions.csv").tail(250)
 
    problem = ntsp_input_from_batch(batch)

    _, _, _, _, _, metrics = solve_ntsp(problem, lambda_weight=0.25, print_console=True)
    
    print(metrics)



    # --- MANY BATCHES ---

    # data = pd.read_csv("settlement_optimization/Data/transactions.csv").tail(12000)
    # batch_metrics = []

    # n_batches = 20
    # b_size = 150

    # for i in range(2, 2+n_batches, 1):
    #     batch = data.iloc[-b_size*i:-b_size*(i-1)]

    #     problem = ntsp_input_from_batch(batch)

    #     _, _, _, _, _, metrics = solve_ntsp(problem, lambda_weight=0.25)

    #     batch_metrics.append(metrics)


    # print(f'Average  {"Settlement Succes"}:  {"{0:0.2f}".format(100*avg_metric(batch_metrics, "settle_rate"))}%')
    # print(f'Average  {"Auto-Collateral used"}:  ${"{0:0.0f}".format(avg_metric(batch_metrics, "total_loan"))}')
    # plot_single_run(batch_metrics)

    
    



