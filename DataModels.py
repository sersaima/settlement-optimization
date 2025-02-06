from typing import List
from pydantic import BaseModel, Field

# ============================================================
# Pydantic Models for Input Data
# ============================================================

class Account(BaseModel):
    """
    Represents an owner's account that holds both cash and security positions.

    Attributes:
      id: Unique account identifier.
      owner_id: Identifier for the owner.
      initial_cash: The cash initially held in the account (a_in^A in Eq. (2a)).
      credit_limit: The maximum additional cash available from collateral.
    """
    id: int = Field(..., description="Account ID.")
    owner_id: int = Field(..., description="Owner ID.")
    initial_cash: float = Field(..., description="Initial cash balance (a_in^A).")
    credit_limit: float = Field(..., description="Credit limit for additional cash (a_m^lim).")


class SecurityPosition(BaseModel):
    """
    Represents a security position held in an account.

    Attributes:
      id: Unique identifier for the security (e.g. ISIN or a numeric code).
      account_id: The account in which this position is held.
      initial_quantity: The initial quantity held (q_in^p in Eq. (2b)).
    """
    id: int = Field(..., description="Security identifier.")
    account_id: int = Field(..., description="ID of the account holding this position.")
    initial_quantity: int = Field(..., description="Initial quantity in the position (q_in^p).")


class Transaction(BaseModel):
    """
    Represents a transaction in the NTSP.

    Attributes:
      id: Unique transaction identifier.
      cash_amount: Cash amount exchanged (aₜ in Eq. (1)).
      weight: Priority weight (wₜ in Eq. (1)).
      debit_account: The account from which cash is debited.
      credit_account: The account to which cash is credited.
      security_id: Identifier of the security traded.
      quantity: Quantity of securities exchanged (qₜ).
      security_flow: Direction of the security flow:
                     -1 indicates securities leave the position,
                     +1 indicates securities enter the position.
    """
    id: int = Field(..., description="Unique transaction ID.")
    cash_amount: float = Field(..., description="Cash amount exchanged (aₜ in Eq. (1)).")
    weight: float = Field(..., description="Transaction weight (wₜ in Eq. (1)).")
    debit_account: int = Field(..., description="ID of the account debited.")
    credit_account: int = Field(..., description="ID of the account credited.")
    security_id: int = Field(..., description="ID of the security traded.")
    quantity: int = Field(..., description="Quantity of securities exchanged (qₜ).")
    security_flow: int = Field(..., description="Flow direction: -1 for outflow, +1 for inflow.")


class CollateralLink(BaseModel):
    """
    Represents a collateral link for auto-collateralization.

    Attributes:
      id: Unique collateral link identifier.
      associated_account: The account (owner) for which this collateral link applies.
      lot_size: Lot size (n) for this link.
      valuation: Valuation per lot (v).
      q_min: Minimum number of lots required (qₗ^min in Eq. (4b)).
      q_lim: Maximum number of lots allowed (qₗ^lim in Eq. (4b)).
      triggered_transactions: List of transaction IDs that trigger activation of this link.
      security_id: Identifier of the security pledged via this link.
    """
    id: int = Field(..., description="Collateral link ID.")
    associated_account: int = Field(..., description="ID of the associated account (owner).")
    lot_size: int = Field(..., description="Lot size (n).")
    valuation: float = Field(..., description="Valuation per lot (v).")
    q_min: int = Field(..., description="Minimum lots (qₗ^min).")
    q_lim: int = Field(..., description="Maximum lots (qₗ^lim).")
    triggered_transactions: List[int] = Field(..., description="List of transaction IDs that trigger this link (T_link).")
    security_id: int = Field(..., description="ID of the security pledged.")


class AfterLink(BaseModel):
    """
    Represents an ordering constraint (after‑link) between two transactions.

    Attributes:
      t1: Transaction ID that must be settled first.
      t2: Transaction ID that can only be settled if t1 is settled (x[t₂] ≤ x[t₁], Eq. (5)).
    """
    t1: int = Field(..., description="ID of the transaction that must be settled first.")
    t2: int = Field(..., description="ID of the dependent transaction (x[t₂] ≤ x[t₁]).")


class NTSPInput(BaseModel):
    """
    Container for all NTSP input data.

    Attributes:
      transactions: List of transactions (T).
      accounts: List of unified accounts (each holding cash and security positions).
      collateral_links: List of collateral links (SPL).
      after_links: List of ordering constraints (Eq. (5)).
      security_positions: List of security positions (each associated with an account).
    """
    transactions: List[Transaction] = Field(..., description="List of transactions (T).")
    accounts: List[Account] = Field(..., description="List of accounts (each owner's portfolio).")
    collateral_links: List[CollateralLink] = Field(..., description="List of collateral links (SPL).")
    after_links: List[AfterLink] = Field(..., description="List of ordering constraints (Eq. (5)).")
    security_positions: List[SecurityPosition] = Field(..., description="List of security positions.")
