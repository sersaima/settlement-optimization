import pandas as pd

# ============================================================
# Code to match the buy and sell orders and generate
# transactions data
# ============================================================

# Read in the raw data
orders_df = pd.read_csv("Trade_Activity_Broadridge_DTCC.csv")
# Only keep trades in USD and sort on timestamp
orders_df = orders_df[orders_df["Settlement_Currency"] == "USD"].sort_values(by=["Execution_Timestamp"])

transactions = []
for instr in set(orders_df["Instrument_Name"]):
    # Match orders per security
    instr_df = orders_df[orders_df["Instrument_Name"] == instr]
    print(instr, len(instr_df))
    buy_queue, sell_queue = [], []
    # Loop through all the orders
    for index, row in instr_df.iterrows():
        quantity = int(row["Quantity"])
        price = row["Execution_Price"]
        trading_account = row["Trading_Account"]
        if row["Direction"] == "Buy":
            # Look for sellers to match with the buyer
            for sell in sell_queue:
                seller = sell["trader"]
                # Avoid that a party transacts with itself
                if seller != trading_account:
                    tx_amount = min(quantity, sell["quantity"])
                    # Store transaction data
                    transactions.append({"timestamp": row["Execution_Timestamp"], "security": instr, "quantity": tx_amount, "price": round((price + sell["price"])/2, 2), "from": seller, "to": trading_account})
                    # Deduct transaction amount from outstanding quantities
                    quantity -= tx_amount
                    sell["quantity"] -= tx_amount
                if quantity <= 0:
                    # If all of the securities of the buyer have been matched, stop looking for sellers
                    break
            sell_queue = [sell for sell in sell_queue if sell["quantity"] > 0]
            if quantity > 0:
                # Put any left-over unmatched amounts in the buy queue
                buy_queue.append({"trader": trading_account, "quantity": quantity, "price": price})
                # Sort buy orders on descending price
                buy_queue.sort(key=lambda x: x["price"], reverse=True)
        elif row["Direction"] == "Sell":
            # Look for buyers to match with the seller
            for buy in buy_queue:
                buyer = buy["trader"]
                # Avoid that a party transacts with itself
                if buyer != trading_account:
                    tx_amount = min(quantity, buy["quantity"])
                    # Store transaction data
                    transactions.append({"timestamp": row["Execution_Timestamp"], "security": instr, "quantity": tx_amount, "price": round((price + buy["price"])/2, 2), "to": buyer, "from": trading_account})
                    # Deduct transaction amount from outstanding quantities
                    quantity -= tx_amount
                    buy["quantity"] -= tx_amount
                if quantity <= 0:
                    # If all of the securities of the seller have been matched, stop looking for buyers
                    break
            buy_queue = [buy for buy in buy_queue if buy["quantity"] > 0]
            if quantity > 0:
                # Put any left-over unmatched amounts in the sell queue
                sell_queue.append({"trader": trading_account, "quantity": quantity, "price": price})
                # Sort sell orders on ascending price
                sell_queue.sort(key=lambda x: x["price"])

# Convert data to dataframe
transactions_df = pd.DataFrame(transactions)
# Sort by timestamp
transactions_df = transactions_df.sort_values(by=["timestamp"])
# Store as csv file
transactions_df.to_csv("transactions.csv", index=False)
