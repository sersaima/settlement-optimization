from Model import solve_ntsp
from DataInput import generate_ntsp_input


if __name__ == "__main__":
    ntsp_data = generate_ntsp_input(
        n_transactions=250,
        n_collateral_links=4,
        n_accounts=10,
        n_securities=20
    )

    solve_ntsp(ntsp_data, lambda_weight=0.5)
