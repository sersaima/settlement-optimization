from DataModels import Account, SecurityPosition, Transaction, CollateralLink, AfterLink, NTSPInput

from typing import List, Tuple, Dict
from ortools.linear_solver import pywraplp

# ============================================================
# Helper Functions
# ============================================================

def get_transaction_lookups(ntsp_data: NTSPInput) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Builds lookup dictionaries for transactions per account.

    Returns:
      T_debit: Maps account IDs to a list of transaction IDs where the account is debited.
      T_credit: Maps account IDs to a list of transaction IDs where the account is credited.
    """
    T_debit = {a.id: [] for a in ntsp_data.accounts}
    T_credit = {a.id: [] for a in ntsp_data.accounts}
    for t in ntsp_data.transactions:
        T_debit[t.debit_account].append(t.id)
        T_credit[t.credit_account].append(t.id)
    return T_debit, T_credit


def get_security_transaction_lookups(ntsp_data: NTSPInput): #-> Tuple[Dict[(int, int), List[Transaction]], Dict[(int, int), List[Transaction]]]:
    """
    Builds lookup dictionaries for transactions per security position.

    For each security (by security id), separates transactions that debit
    (security_flow == -1) and credit (security_flow == +1) the security position.

    Returns:
      T_debit_sec: Maps security IDs to a list of transactions with security_flow == -1.
      T_credit_sec: Maps security IDs to a list of transactions with security_flow == +1.
    """
    T_debit_sec: Dict[(int, int), List[Transaction]] = {}
    T_credit_sec: Dict[(int, int), List[Transaction]] = {}

    for sp in ntsp_data.security_positions:
        T_debit_sec[(sp.account_id, sp.id)] = []
        T_credit_sec[(sp.account_id, sp.id)] = []

    for t in ntsp_data.transactions:
        if (t.credit_account, t.security_id) in T_debit_sec:
            T_debit_sec[(t.credit_account, t.security_id)].append(t)

        if (t.debit_account, t.security_id) in T_credit_sec:
            T_credit_sec[(t.debit_account, t.security_id)].append(t)

    return T_debit_sec, T_credit_sec



# ============================================================
# Model Building Functions (One per Equation/Set)
# ============================================================

def build_decision_variables(solver, ntsp_data: NTSPInput):
    """
    Builds decision variables.

    Returns:
      x: Dict mapping transaction IDs to binary variables (settlement decision; Eq. (1)).
      y: Dict mapping collateral link IDs to integer variables (number of lots pledged; Eq. (4b)).
      z: Dict mapping collateral link IDs to binary activation variables.
      need: Dict mapping account IDs to continuous variables representing extra cash needed.
    """
    x = {t.id: solver.BoolVar(f"x_{t.id}") for t in ntsp_data.transactions}
    y = {l.id: solver.IntVar(0, l.q_lim, f"y_{l.id}") for l in ntsp_data.collateral_links}
    z = {l.id: solver.BoolVar(f"z_{l.id}") for l in ntsp_data.collateral_links}

    need = {}
    for acc in ntsp_data.accounts:
        # Upper bound: sum of cash amounts for transactions debiting this account.
        ub = sum(t.cash_amount for t in ntsp_data.transactions if t.debit_account == acc.id)
        need[acc.id] = solver.NumVar(0, ub, f"need_{acc.id}")
    return x, y, z, need


def add_objective(solver, ntsp_data: NTSPInput, x: Dict[int, any], lambda_weight: float):
    """
    Adds the objective function (Eq. (1)).

    Maximize: λ * ∑ₜ (wₜ * aₜ * xₜ) + (1-λ) * ∑ₜ (wₜ * xₜ).
    (Constant denominators are omitted.)
    """
    solver.Maximize(
        lambda_weight * solver.Sum([t.weight * t.cash_amount * x[t.id] for t in ntsp_data.transactions]) * (1 / sum([t.weight * t.cash_amount for t in ntsp_data.transactions]))
        + (1 - lambda_weight) * solver.Sum([t.weight * x[t.id] for t in ntsp_data.transactions]) * (1 / sum([t.weight for t in ntsp_data.transactions]))
    )


def add_account_constraints(solver, ntsp_data: NTSPInput, x: Dict[int, any],
                            need: Dict[int, any], y: Dict[int, any]):
    """
    Adds account cash constraints corresponding to Eq. (2a)/(4a).

    For each Account A:
      Let T_debit(A) and T_credit(A) be the transactions debiting and crediting A.
      Enforce:
          (∑ₜ∈T_debit(A) aₜ·xₜ - ∑ₜ∈T_credit(A) aₜ·xₜ) ≤ initial_cash + collateral_expr,
      where collateral_expr = ∑_{l with associated_account == A.id} (lot_size * valuation * y[l]),
      and also ensure that collateral_expr ≤ credit_limit.
    Also, define:
          need[A] ≥ (∑ₜ∈T_debit(A) aₜ·xₜ - ∑ₜ∈T_credit(A) aₜ·xₜ - initial_cash).
    """
    T_debit, T_credit = get_transaction_lookups(ntsp_data)
    transactions_dict = {t.id: t for t in ntsp_data.transactions}
    for acc in ntsp_data.accounts:
        debit_sum = solver.Sum([transactions_dict[t_id].cash_amount * x[t_id] for t_id in T_debit[acc.id]])
        credit_sum = solver.Sum([transactions_dict[t_id].cash_amount * x[t_id] for t_id in T_credit[acc.id]])
        solver.Add(need[acc.id] >= debit_sum - credit_sum - acc.initial_cash)
        collateral_expr = solver.Sum([
            l.lot_size * l.valuation * y[l.id]
            for l in ntsp_data.collateral_links if l.associated_account == acc.id
        ])
        solver.Add(debit_sum - credit_sum <= acc.initial_cash + collateral_expr)
        solver.Add(collateral_expr <= acc.credit_limit)


def add_after_link_constraints(solver, ntsp_data: NTSPInput, x: Dict[int, any]):
    """
    Adds after-link (ordering) constraints (Eq. (5)).

    For each after-link (t₁, t₂): enforce x[t₂] ≤ x[t₁].
    """
    for al in ntsp_data.after_links:
        solver.Add(x[al.t2] <= x[al.t1])


def add_collateral_link_constraints(solver, ntsp_data: NTSPInput,
                                    x: Dict[int, any], y: Dict[int, any], z: Dict[int, any]):
    """
    Adds collateral link activation and bounds constraints (Eq. (4b)).

    For each collateral link l:
      - Activation: For each triggered transaction t, enforce x[t] ≤ z[l]. Also, enforce:
             z[l] ≤ ∑ (x[t] for t in triggered_transactions).
      - Bounds: When activated (z[l]=1), enforce: q_min ≤ n * y[l] ≤ q_lim.
    """
    for l in ntsp_data.collateral_links:
        for t_id in l.triggered_transactions:
            solver.Add(x[t_id] <= z[l.id])
        solver.Add(z[l.id] <= solver.Sum([x[t_id] for t_id in l.triggered_transactions]))
        solver.Add(l.lot_size * y[l.id] >= l.q_min * z[l.id])
        solver.Add(l.lot_size * y[l.id] <= l.q_lim * z[l.id])


def add_security_position_constraints(solver, ntsp_data: NTSPInput, x: Dict[int, any], y: Dict[int, any]):
    """
    Adds security position constraints (Eq. (2b)).

    For each SecurityPosition p (held by an Account):
      Let T_debit_p be the transactions with t.security_id == p.id and t.security_flow == -1.
      Let T_credit_p be the transactions with t.security_id == p.id and t.security_flow == +1.
      Define collateralized quantity for position p as:
            q_col^p = ∑_{l with l.associated_account == p.account_id and l.security_id == p.id} (lot_size * y[l]).
      Then enforce:
            ∑ₜ∈T_debit_p (qₜ * xₜ) - ∑ₜ∈T_credit_p (qₜ * xₜ) ≤ p.initial_quantity - q_col^p.
    """
    T_debit_sec, T_credit_sec = get_security_transaction_lookups(ntsp_data)
    for sp in ntsp_data.security_positions:
        # print(f'Security Position:\n\t sec_id: {sp.id}\n\t owner_ir: {sp.account_id}')
        debit_qty = solver.Sum([t.quantity * x[t.id] for t in T_debit_sec.get((sp.account_id, sp.id), [])])
        credit_qty = solver.Sum([t.quantity * x[t.id] for t in T_credit_sec.get((sp.account_id, sp.id), [])])
        collateral_qty = solver.Sum([
            l.lot_size * y[l.id]
            for l in ntsp_data.collateral_links
            if l.associated_account == sp.account_id and l.security_id == sp.id
        ])
        solver.Add(debit_qty - credit_qty <= sp.initial_quantity - collateral_qty)


# ============================================================
# Main Model Solver Function
# ============================================================

def solve_ntsp(ntsp_data: NTSPInput, lambda_weight: float = 0.5, print_console = False):
    """
    Builds and solves the complete NTSP model.

    This function creates the OR-Tools solver, builds decision variables,
    adds the objective and all constraints (account, after-links, collateral links,
    and security positions), solves the model, and prints the solution.

    Args:
      ntsp_data: NTSPInput instance containing all input data.
      lambda_weight: Weighting factor λ ∈ [0,1] for the objective (Eq. (1)).
    """
    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        print("Error: Could not create solver.")
        return

    # Build decision variables.
    x, y, z, need = build_decision_variables(solver, ntsp_data)

    # Add the objective function (Eq. (1)).
    add_objective(solver, ntsp_data, x, lambda_weight)

    # Add account constraints (Eqs. (2a) and (4a)).
    add_account_constraints(solver, ntsp_data, x, need, y)

    # Add after-link ordering constraints (Eq. (5)).
    add_after_link_constraints(solver, ntsp_data, x)

    # Add collateral link constraints (Eq. (4b)).
    add_collateral_link_constraints(solver, ntsp_data, x, y, z)

    # Add security position constraints (Eq. (2b)).
    add_security_position_constraints(solver, ntsp_data, x, y)



    # Solve the model.
    status = solver.Solve()
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        metrics = { }
        metrics["total_cashflow"] = 0
        metrics["settle_rate"] = 0
        for t in ntsp_data.transactions:
            metrics["total_cashflow"] += int(x[t.id].solution_value()) * t.cash_amount
            metrics["settle_rate"] += int(x[t.id].solution_value())

        metrics["settle_rate"] = metrics["settle_rate"] / len(ntsp_data.transactions)

        metrics["total_loan"] = 0
        for sp in ntsp_data.security_positions:
            collateral_total = sum(l.lot_size * y[l.id].solution_value()
                                    for l in ntsp_data.collateral_links
                                    if l.associated_account == sp.account_id and l.security_id == sp.id)
            metrics["total_loan"] += collateral_total


        if print_console:
            print("Solution Found!")
            print("Objective Value =", solver.Objective().Value())
            print("\nTransaction Settlement Decisions:")
            for t in ntsp_data.transactions:
                print(f"  Transaction {t.id}: x = {'Success' if int(x[t.id].solution_value()) else 'Failed'}")

            # print("\nAccount Cash Status (need variables):")
            # for acc in ntsp_data.accounts:
            #     print(f"  Account {acc.id} (Owner {acc.owner_id}): need = {need[acc.id].solution_value()}")

            print("\nCollateral Link Decisions:")
            for l in ntsp_data.collateral_links:
                if int(z[l.id].solution_value()):
                    print(f"  Collateral Link {l.id} (Security {l.security_id}): Lots amount (y) = {y[l.id].solution_value()}, "
                        f"For account {l.associated_account}. \tUsed in {len(l.triggered_transactions)} transactions.")

            print("\nSecurity Position Summary:")
            T_debit_sec, T_credit_sec = get_security_transaction_lookups(ntsp_data)
            for sp in ntsp_data.security_positions:
                debit_total = sum(t.quantity * x[t.id].solution_value() for t in T_debit_sec.get((sp.account_id, sp.id), []))
                credit_total = sum(t.quantity * x[t.id].solution_value() for t in T_credit_sec.get((sp.account_id, sp.id), []))
                collateral_total = sum(l.lot_size * y[l.id].solution_value()
                                        for l in ntsp_data.collateral_links
                                        if l.associated_account == sp.account_id and l.security_id == sp.id)
                net_flow = debit_total - credit_total
                available = sp.initial_quantity - collateral_total
                if net_flow != 0:
                    print(f"  Security Position {sp.id} (Account {sp.account_id}): Net outflow = {net_flow} ≤ {available}")

        return solver, x, y, z, need, metrics
    else:
        print("No feasible solution found.")
