from DataModels import Account, SecurityPosition, Transaction, CollateralLink, AfterLink, NTSPInput

import random

import numpy as np
import pandas as pd



def ntsp_input_from_batch(batch):
    accs, sec_poss, coll_links = _gen_init_conditions(batch)
    trans, after_links = _transactions_from_csv(batch)

    for cl in coll_links:
        cl.triggered_transactions = [t.id for t in trans if t.security_id == cl.security_id]

    return NTSPInput(
        transactions=trans,
        accounts=accs,
        collateral_links=coll_links,
        after_links=after_links,
        security_positions=sec_poss
    )




def _gen_init_conditions(batch):
    all_securities = batch["security"].unique()
    all_parties = pd.concat([batch["to"], batch["from"]]).unique()

    cash_accounts = []
    security_positions = []
    sec_collateral_links = []

    cli = 1
    for party in all_parties:
        party_buys = batch.loc[batch["to"] == party]
        avg_cash_sell = np.mean(np.multiply(party_buys["price"], party_buys["quantity"]))

        init_cash = max(25000, 2.5*avg_cash_sell + np.random.normal(1.5*avg_cash_sell, 1.5*avg_cash_sell))
        cash_credit_limit = max(25000, 2*init_cash + np.random.normal(init_cash, 1.5*init_cash))

        p_acc = Account(
            id=hash(party),
            owner_id=hash(party),  # For simplicity, we use the same value.
            initial_cash=int(np.floor(init_cash)),
            credit_limit=int(np.floor(cash_credit_limit))
            )
        cash_accounts.append(p_acc)

        
        for security in all_securities:
            party_sells = batch.loc[(batch["from"] == party) & (batch["security"] == security)]

            if len(party_sells.index) > 0:
                avg_sec_sell = np.mean(party_sells["quantity"])
                init_sec = max(10000, 2.5*avg_sec_sell + np.random.normal(1.5*avg_sec_sell, 1.5*avg_sec_sell))
            else:
                init_sec = 10000 * int(np.random.random() > 0.75)
            
            p_sec_pos = SecurityPosition(
                id=hash(security),
                account_id=hash(party),
                initial_quantity=int(np.floor(init_sec))
            )
            security_positions.append(p_sec_pos)

            if (len(party_sells.index) > 0) and (avg_sec_sell >= 0.5*np.mean(batch.loc[batch["from"] == party]["quantity"])) and (np.random.random() > 0.2):
                c_link_sec_party = CollateralLink(
                    id=cli,
                    associated_account=hash(party),
                    lot_size=500,
                    valuation=0.95*np.mean(party_sells["price"]),
                    q_min=500,
                    q_lim=int(np.floor((0.5 + np.random.random())*init_sec*2)),
                    triggered_transactions=[],  # to be set later.
                    security_id=hash(security)
                )
                cli = cli + 1
                sec_collateral_links.append(c_link_sec_party)

    return cash_accounts, security_positions, sec_collateral_links


def _transactions_from_csv(batch):
    transactions = []

    for index, row in batch.iterrows():
        cash_amount = row["price"] * row["quantity"]

        priority = 0.5*abs(np.log(np.power(cash_amount, 0.25) * len(batch.loc[(batch["from"] == row["from"]) | 
                                                        (batch["from"] == row["to"]) | 
                                                        (batch["to"] == row["from"]) | 
                                                        (batch["to"] == row["to"])].index)))

        trans = Transaction(
            id=index,
            cash_amount=cash_amount,
            weight=priority,
            debit_account=hash(row["to"]),
            credit_account=hash(row["from"]),
            security_id=hash(row["security"]),
            quantity=int(row["quantity"]),
        )
        transactions.append(trans)
    
    after_links = []
    all_parties = pd.concat([batch["to"], batch["from"]]).unique()

    for party in all_parties:
        party_transactions = batch.loc[(batch["from"] == party) | (batch["to"] == party)]
        if (len(party_transactions.index) > 0.05*len(batch.index)) and (len(party_transactions.index) > 6) and (np.random.rand() > 0.75):
            for k in [0, 2, 4]:
                linkk = AfterLink(
                    t1=party_transactions.iloc[k].name,
                    t2=party_transactions.iloc[k+1].name)
                after_links.append(linkk)


    return transactions, after_links



def generate_ntsp_input(
    n_transactions: int,
    n_collateral_links: int,
    n_accounts: int,
    n_securities: int
) -> NTSPInput:
    """
    Generates an NTSPInput instance with:
      - n_transactions transactions
      - n_collateral_links collateral links
      - n_accounts accounts (each representing an owner's portfolio)
      - n_securities distinct securities

    The function creates:
      1. A list of distinct security IDs (starting at 101).
      2. A list of Accounts, each with random initial cash and credit limit.
      3. A list of SecurityPositions, assigning each distinct security to a random account.
      4. A list of Transactions:
           - Each transaction randomly selects a debit and a credit account (ensuring they differ).
           - A random security is chosen from the distinct set.
           - A random cash amount and weight are generated.
           - A random quantity is generated and the security_flow is set to -1 (indicating an outflow).
      5. A list of CollateralLinks:
           - Each link is associated with a random account.
           - Random values for lot_size, valuation, q_min, and q_lim are generated.
           - A random security is chosen.
      6. After-link Constraints:
           - For approximately 30% of transactions (with id > 0), an after-link constraint is created.
      7. Finally, for each collateral link, its triggered_transactions are set to all transactions that use the same security.

    Returns:
        A valid NTSPInput instance.
    """
    # random.seed(42)  # For reproducibility

    # 1) Define a list of distinct security IDs.
    securities = list(range(101, 101 + n_securities))

    # 2) Generate Accounts.
    accounts = []
    for i in range(n_accounts):
        init_cash = random.randint(500, 1000)
        credit_limit = init_cash + random.randint(500, 2000)
        accounts.append(Account(
            id=i,
            owner_id=i,  # For simplicity, we use the same value.
            initial_cash=init_cash,
            credit_limit=credit_limit
        ))

    # 3) Generate SecurityPositions.
    # Ensure each distinct security is held by at least one account.
    security_positions = []
    for sec in securities:
        for acc_id in random.choices([acc.id for acc in accounts], k = random.randint(2, n_accounts // 2)):
            # Randomly assign this security to one of the accounts.
            init_qty = random.randint(100, 1000)
            security_positions.append(SecurityPosition(
                id=sec,
                account_id=acc_id,
                initial_quantity=init_qty
            ))

    # 4) Generate Transactions.
    transactions = []
    for i in range(n_transactions):
        cash_amount = random.randint(100, 2000)
        weight = round(random.uniform(0.5, 2.0), 2)
        # Choose debit and credit accounts ensuring they are different.
        debit_acc = random.choice([acc.id for acc in accounts])
        possible_credit = [acc.id for acc in accounts if acc.id != debit_acc]
        credit_acc = random.choice(possible_credit) if possible_credit else debit_acc
        sec_id = random.choice(securities)
        # For simplicity, all transactions here reduce the security position (outflow).
        quantity = random.randint(10, 100)
        #security_flow = -1
        transactions.append(Transaction(
            id=i,
            cash_amount=cash_amount,
            weight=weight,
            debit_account=debit_acc,
            credit_account=credit_acc,
            security_id=sec_id,
            quantity=quantity,
           # security_flow=security_flow
        ))

    # Ensure each security is used by at least one transaction.
    used_securities_in_tx = {t.security_id for t in transactions}
    for sec in securities:
        if sec not in used_securities_in_tx:
            idx = random.randint(0, n_transactions - 1)
            transactions[idx].security_id = sec
            used_securities_in_tx.add(sec)

    # 5) Generate CollateralLinks.
    collateral_links = []
    for i in range(n_collateral_links):
        associated_acc = random.choice([acc.id for acc in accounts])
        lot_size = random.randint(1, 3)
        valuation = random.randint(50, 200)
        q_min = random.randint(1, 3)
        q_lim = q_min + random.randint(1, 4)
        sec_id = random.choice(securities)
        collateral_links.append(CollateralLink(
            id=i,
            associated_account=associated_acc,
            lot_size=lot_size,
            valuation=valuation,
            q_min=q_min,
            q_lim=q_lim,
            triggered_transactions=[],  # to be set later.
            security_id=sec_id
        ))

    # Ensure each security is used by at least one collateral link.
    used_securities_in_links = {cl.security_id for cl in collateral_links}
    for sec in securities:
        if sec not in used_securities_in_links and collateral_links:
            idx = random.randint(0, n_collateral_links - 1)
            collateral_links[idx].security_id = sec
            used_securities_in_links.add(sec)

    # 6) Generate After-Link Constraints (approximately 30% of transactions, except the first).
    after_links = []
    for t in transactions:
        if t.id > 0 and random.random() < 0.3:
            possible_preds = [u.id for u in transactions if u.id < t.id]
            if possible_preds:
                t1 = random.choice(possible_preds)
                after_links.append(AfterLink(t1=t1, t2=t.id))

    # 7) Set triggered_transactions for each collateral link:
    # For each collateral link, include all transactions that share the same security.
    for cl in collateral_links:
        cl.triggered_transactions = [t.id for t in transactions if t.security_id == cl.security_id]

    return NTSPInput(
        transactions=transactions,
        accounts=accounts,
        collateral_links=collateral_links,
        after_links=after_links,
        security_positions=security_positions
    )
