import os  # ensure available for helpers
from git.repo import Repo  # ensure available for helpers
import streamlit as st  # ensure available for helpers

# New: auto commit-and-push helper for specific files
def commit_and_push_files(file_paths: list[str], message: str) -> None:
    try:
        repo = Repo(REPO_PATH)
    except Exception:
        st.warning("Git: Not a repository. Changes saved locally; auto-push disabled. Initialize Git to enable auto-push.")
        return

    try:
        for p in file_paths:
            if os.path.exists(p):
                repo.git.add(p)
        if repo.index.diff("HEAD") or repo.untracked_files:
            repo.index.commit(message)
        try:
            origin = repo.remote(name='origin')
        except ValueError:
            st.warning("Git: Remote 'origin' not found. Changes committed locally; set up a remote to push.")
            return
        # Determine active branch if possible
        try:
            branch_name = repo.active_branch.name
        except Exception:
            branch_name = None

        try:
            # Use explicit refspec when we know the branch
            if branch_name:
                origin.push(refspec=f"HEAD:{branch_name}")
            else:
                origin.push()
            st.toast("Auto-pushed to GitHub")
        except Exception as e:
            # Attempt to resolve non-fast-forward by pulling with rebase, then push again
            try:
                if branch_name:
                    st.info("Git: Push failed, attempting pull --rebase from origin/{branch} then retry push...".format(branch=branch_name))
                    repo.git.pull('origin', branch_name, '--rebase')
                    origin.push(refspec=f"HEAD:{branch_name}")
                else:
                    st.info("Git: Push failed, attempting pull --rebase then retry push...")
                    repo.git.pull('--rebase')
                    origin.push()
                st.toast("Auto-pushed to GitHub after rebase")
            except Exception as e2:
                st.warning("Git push failed after rebase. Changes committed locally. Please ensure:\n"
                           "- You have permission to push to the repository.\n"
                           "- Your Git credentials are configured on this machine.\n"
                           "- The current branch tracks origin and has no unresolved conflicts.\n"
                           f"Details: {e2}")
    except Exception as e:
        st.warning(f"Git operation issue. Changes saved locally. Details: {e}")

def write_csv_atomic(df, file_path: str) -> None:
    """Write DataFrame to CSV file atomically"""
    temp_path = file_path + '.tmp'
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, file_path)
    # If a key CSV was updated, refresh docs immediately and push
    try:
        abs_path = os.path.abspath(file_path)
        if abs_path in (
            os.path.abspath(CLIENT_EXPENSES_FILE),
            os.path.abspath(CSV_FILE),
            os.path.abspath(PEOPLE_FILE),
        ):
            changed = regenerate_docs_index_html()
            try:
                git_push()
            except Exception:
                pass
    except Exception:
        # Never block saving due to docs update issues
        pass

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import git
from git.repo import Repo
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import shutil
import zipfile
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and Configuration


class Config:
    """Configuration class containing all constants for the application."""

    # Valid status values
    VALID_CHEQUE_STATUSES: List[str] = [
        "received/given", "processing", "bounced", "processing done"]
    VALID_TRANSACTION_STATUSES: List[str] = ["completed", "pending"]
    VALID_EXPENSE_CATEGORIES: List[str] = [
        "Labour", "Material", "Travel", "Food & Beverages", "Accomodation", "General"]

    # Column definitions for each data type
    PAYMENT_COLUMNS: List[str] = [
        "payment_method", "cheque_status", "transaction_status",
        "reference_number", "date", "amount", "person",
        "type", "status", "description"
    ]

    CLIENT_EXPENSE_COLUMNS: List[str] = [
        "original_transaction_ref_num", "expense_date", "expense_person",
        "expense_category", "expense_amount", "expense_quantity",
        "expense_description"
    ]

# Data cleaners
class DataCleaner:
    """Normalize and validate payments data for consistency and safe writes."""
    @staticmethod
    def clean_payments_data(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            # Ensure required columns exist even for empty frames
            return pd.DataFrame(columns=Config.PAYMENT_COLUMNS)

        df = df.copy()
        # Ensure all required columns exist
        for col in Config.PAYMENT_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Normalize reference_number to clean string
        def _norm_ref(x):
            s = "" if pd.isna(x) else str(x).strip()
            if s.lower() in ("nan", "none", "null"):  # treat as empty
                return ""
            return s

        df['reference_number'] = df['reference_number'].apply(_norm_ref)

        # Coerce amount numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)

        # Lowercase enums safely
        for col in ['payment_method', 'type', 'status', 'transaction_status', 'cheque_status']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        # Date normalization (keep as datetime)
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except Exception:
            pass

        # Reorder columns
        df = df[Config.PAYMENT_COLUMNS]
        return df


class ExpenseCleaner:
    """Normalize and validate client expenses data."""
    @staticmethod
    def clean_client_expenses_data(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=Config.CLIENT_EXPENSE_COLUMNS)

        df = df.copy()
        # Ensure all required columns exist
        for col in Config.CLIENT_EXPENSE_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Coerce numerics
        df['expense_amount'] = pd.to_numeric(df['expense_amount'], errors='coerce').fillna(0.0)
        df['expense_quantity'] = pd.to_numeric(df['expense_quantity'], errors='coerce').fillna(1.0)

        # Normalize strings
        for col in ['expense_person', 'expense_category', 'expense_description', 'original_transaction_ref_num']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Date normalization
        try:
            df['expense_date'] = pd.to_datetime(df['expense_date'], errors='coerce')
        except Exception:
            pass

        # Reorder
        df = df[Config.CLIENT_EXPENSE_COLUMNS]
        return df

# Convenience constants and function aliases used across the UI
# These prevent NameError and centralize allowed values
valid_cheque_statuses_lower: List[str] = [s.lower()
                                          for s in Config.VALID_CHEQUE_STATUSES]
valid_transaction_statuses_lower: List[str] = [
    s.lower() for s in Config.VALID_TRANSACTION_STATUSES]
valid_expense_categories: List[str] = Config.VALID_EXPENSE_CATEGORIES

# Aliases for cleaner calls in the UI layer
clean_payments_data = DataCleaner.clean_payments_data
clean_client_expenses_data = ExpenseCleaner.clean_client_expenses_data

# Initialize session state for all variables used in the app


def init_state():
    keys = [
        'selected_transaction_type', 'payment_method', 'editing_row_idx', 'selected_person', 'reset_add_form',
        'add_amount', 'add_date', 'add_reference_number', 'add_cheque_status', 'add_status', 'add_description',
        'temp_edit_data', 'invoice_person_name', 'invoice_type', 'invoice_start_date', 'invoice_end_date',
        'generated_invoice_pdf_path', 'show_download_button',
        'view_person_filter', 'view_reference_number_search', 'view_date_filter_start', 'view_date_filter_end',
        'view_type_filter', 'view_method_filter', 'view_status_filter',
        'selected_client_for_expense', 'add_client_expense_amount', 'add_client_expense_date',
        'add_client_expense_category', 'add_client_expense_description', 'reset_client_expense_form',
        'add_client_expense_quantity',
        'report_person_name', 'report_start_date', 'report_end_date', 'report_type',
        'editing_expense_row_idx', 'temp_edit_expense_data', 'selected_expense_to_edit',
        'expense_person_filter', 'expense_date_filter_start', 'expense_date_filter_end',
        'expense_category_filter', 'backup_created_path'
    ]
    defaults = {
        'selected_transaction_type': 'Paid to Me',
        'payment_method': 'cash',
        'editing_row_idx': None,
        'selected_person': "Select...",
        'reset_add_form': False,
        'add_amount': None,
        'add_date': datetime.now().date(),
        'add_reference_number': '',
        'add_cheque_status': 'received/given',
        'add_status': 'completed',
        'add_description': '',
        'temp_edit_data': {},
        'invoice_person_name': "Select...",
        'invoice_type': 'Invoice for Person (All Transactions)',
        'invoice_start_date': datetime.now().date().replace(day=1),
        'invoice_end_date': datetime.now().date(),
        'generated_invoice_pdf_path': None,
        'show_download_button': False,
        'view_person_filter': "All",
        'view_reference_number_search': "",
        'view_date_filter_start': datetime.now().date().replace(day=1),
        'view_date_filter_end': datetime.now().date(),
        'view_type_filter': "All",
        'view_method_filter': "All",
        'view_status_filter': "All",
        'selected_client_for_expense': "Select...",
        'add_client_expense_amount': None,
        'add_client_expense_date': datetime.now().date(),
        'add_client_expense_category': 'General',
        'add_client_expense_description': '',
        'reset_client_expense_form': False,
        'add_client_expense_quantity': 1.0,
        'report_person_name': "Select...",
        'report_start_date': datetime.now().date().replace(day=1),
        'report_end_date': datetime.now().date(),
        'report_type': 'Inquiry',
        'editing_expense_row_idx': None,
        'temp_edit_expense_data': {},
        'selected_expense_to_edit': "Select an expense",
        'expense_person_filter': "All",
        'expense_date_filter_start': datetime.now().date().replace(day=1),
        'expense_date_filter_end': datetime.now().date(),
        'expense_category_filter': "All",
        'backup_created_path': None,
        'custom_expense_categories': [],
        'show_add_category_input': False
    }
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = defaults[k]


init_state()

# Define file paths
REPO_PATH = str(Path(__file__).resolve().parent)
CSV_FILE = os.path.join(REPO_PATH, "payments.csv")
PEOPLE_FILE = os.path.join(REPO_PATH, "people.csv")
CLIENT_EXPENSES_FILE = os.path.join(REPO_PATH, "client_expenses.csv")
SUMMARY_FILE = os.path.join(REPO_PATH, "docs/index.html")
SUMMARY_URL = "https://workprofile-web.github.io/payments-summary/"
INVOICE_DIR = os.path.join(REPO_PATH, "docs", "invoices")

# Utility: auto-update public HTML summary when data changes
import re

def _read_text(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''

def _write_text(path: str, content: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def regenerate_docs_index_html() -> bool:
    """Regenerate key values inside docs/index.html from current CSV data.
    Returns True if file updated, False if no change or on failure.
    """
    try:
        html = _read_text(SUMMARY_FILE)
        if not html:
            return False

        # Compute totals from data
        total_received = 0.0
        total_paid = 0.0
        total_client_expenses = 0.0

        try:
            pay_df = pd.read_csv(CSV_FILE, keep_default_na=False)
            # Minimal normalization to avoid dependency on cleaners during docs rebuild
            if 'type' in pay_df.columns:
                pay_df['type'] = pay_df['type'].astype(str).str.strip().str.lower()
            if 'amount' in pay_df.columns:
                pay_df['amount'] = pd.to_numeric(pay_df['amount'], errors='coerce').fillna(0.0)
            total_received = pay_df[pay_df.get('type', pd.Series(dtype=str)) == 'paid_to_me'].get('amount', pd.Series(dtype=float)).sum()
            total_paid = pay_df[pay_df.get('type', pd.Series(dtype=str)) == 'i_paid'].get('amount', pd.Series(dtype=float)).sum()
        except Exception:
            pass

        try:
            exp_df = pd.read_csv(CLIENT_EXPENSES_FILE, keep_default_na=False)
            # Minimal normalization
            exp_df['expense_amount'] = pd.to_numeric(exp_df.get('expense_amount'), errors='coerce').fillna(0.0)
            exp_df['expense_quantity'] = pd.to_numeric(exp_df.get('expense_quantity'), errors='coerce').fillna(1.0)
            exp_df['line_total'] = exp_df['expense_amount'] * exp_df['expense_quantity']
            total_client_expenses = float(exp_df['line_total'].sum())
        except Exception:
            pass

        # Helper to replace card amount given the card title
        def replace_card_amount(doc: str, title: str, amount: float) -> str:
            pattern = (
                r"(<div class=\"card-title\">" + re.escape(title) + r"</div>\s*<div class=\"card-amount\">)Rs\.[^<]*(</div>)"
            )
            return re.sub(pattern, lambda m: f"{m.group(1)}Rs. {amount:,.2f}{m.group(2)}", doc, flags=re.DOTALL)

        updated = html
        updated = replace_card_amount(updated, "Total Received", float(total_received))
        updated = replace_card_amount(updated, "Total Paid", float(total_paid))
        updated = replace_card_amount(updated, "Total Client Expenses", float(total_client_expenses))

        # Also update the standalone "Total Client Expenses:" summary line if present
        updated = re.sub(
            r"(Total Client Expenses:\s*Rs\.\s*)[\d,]+\.?\d*",
            lambda m: f"{m.group(1)}{total_client_expenses:,.2f}",
            updated
        )

        # Update person options in the filter dropdown
        try:
            ppl_df = pd.read_csv(PEOPLE_FILE, keep_default_na=False)
            names = (ppl_df.get('name') or pd.Series(dtype=str)).astype(str).str.strip()
            options_html = ''.join([f'<option value="{n}">{n}</option>' for n in names if n])
            # Replace the options inside the select with id="name-filter"
            updated = re.sub(
                r"(<select id=\"name-filter\">\s*<option value=\"\">All</option>)(.*?)(</select>)",
                lambda m: m.group(1) + options_html + m.group(3),
                updated,
                flags=re.DOTALL
            )
        except Exception:
            pass

        # Rebuild Client Expenses tables (summary and detailed) from CSV
        try:
            # Detailed client expenses table rows
            exp_df2 = pd.read_csv(CLIENT_EXPENSES_FILE, keep_default_na=False)
            exp_df2['expense_date'] = pd.to_datetime(exp_df2.get('expense_date'), errors='coerce')
            exp_df2['expense_person'] = exp_df2.get('expense_person').astype(str)
            exp_df2['expense_category'] = exp_df2.get('expense_category').astype(str)
            exp_df2['expense_amount'] = pd.to_numeric(exp_df2.get('expense_amount'), errors='coerce').fillna(0.0)
            exp_df2['expense_quantity'] = pd.to_numeric(exp_df2.get('expense_quantity'), errors='coerce').fillna(1.0)
            exp_df2['expense_description'] = exp_df2.get('expense_description').astype(str)
            exp_df2['line_total'] = exp_df2['expense_amount'] * exp_df2['expense_quantity']

            detailed_rows = []
            for row in exp_df2.sort_values('expense_date').itertuples(index=False):
                d = getattr(row, 'expense_date')
                date_str = d.strftime('%Y-%m-%d') if pd.notna(d) else ''
                person = getattr(row, 'expense_person')
                category = getattr(row, 'expense_category') if hasattr(row, 'expense_category') else ''
                amount_unit = float(getattr(row, 'expense_amount') or 0.0)
                qty_val = float(getattr(row, 'expense_quantity') or 0.0)
                qty_str = str(int(qty_val)) if qty_val.is_integer() else f"{qty_val}"
                line_total = float(getattr(row, 'line_total') or 0.0)
                desc = getattr(row, 'expense_description')
                detailed_rows.append(
                    f"<tr data-date=\"{date_str}\" data-person=\"{person}\" data-category=\"{category}\" data-amount=\"{amount_unit}\" data-qty=\"{qty_val}\" data-total=\"{line_total}\">\n"
                    f"    <td>{date_str}</td>\n"
                    f"    <td>{person}</td>\n"
                    f"    <td>{category}</td>\n"
                    f"    <td>Rs. {amount_unit:,.2f}</td>\n"
                    f"    <td>{qty_str}</td>\n"
                    f"    <td>Rs. {line_total:,.2f}</td>\n"
                    f"    <td>{desc}</td>\n"
                    f"</tr>"
                )
            detailed_html = "\n".join(detailed_rows)

            # Replace detailed expenses tbody
            updated = re.sub(
                r"(<table class=\"detailed-expenses-table\">[\s\S]*?<tbody>)([\s\S]*?)(</tbody>)",
                lambda m: m.group(1) + "\n" + detailed_html + "\n" + m.group(3),
                updated,
                flags=re.DOTALL
            )

            # Summary per client: Paid to client (I Paid) vs Spent by client
            paid_df = pd.read_csv(CSV_FILE, keep_default_na=False)
            if 'type' in paid_df.columns:
                paid_df['type'] = paid_df['type'].astype(str).str.strip().str.lower()
            paid_df['amount'] = pd.to_numeric(paid_df.get('amount'), errors='coerce').fillna(0.0)
            paid_df['person'] = paid_df.get('person').astype(str)

            paid_to_client = paid_df[paid_df.get('type') == 'i_paid'].groupby('person', dropna=False)['amount'].sum()
            spent_by_client = exp_df2.groupby('expense_person', dropna=False)['line_total'].sum()

            persons = sorted(set(paid_to_client.index.astype(str)).union(set(spent_by_client.index.astype(str))))
            summary_rows = []
            for p in persons:
                paid_val = float(paid_to_client.get(p, 0.0))
                spent_val = float(spent_by_client.get(p, 0.0))
                remaining = paid_val - spent_val
                rem_class = 'negative-balance' if remaining < 0 else 'positive-balance'
                summary_rows.append(
                    f"<tr>\n"
                    f"    <td>{p}</td>\n"
                    f"    <td>Rs. {paid_val:,.2f}</td>\n"
                    f"    <td>Rs. {spent_val:,.2f}</td>\n"
                    f"    <td class=\"{rem_class}\">Rs. {remaining:,.2f}</td>\n"
                    f"</tr>"
                )
            summary_html = "\n".join(summary_rows)

            # Replace client-summary-table tbody
            updated = re.sub(
                r"(<table class=\"client-summary-table\">[\s\S]*?<tbody>)([\s\S]*?)(</tbody>)",
                lambda m: m.group(1) + "\n" + summary_html + "\n" + m.group(3),
                updated,
                flags=re.DOTALL
            )
        except Exception:
            pass

        # Rebuild All Transactions table rows and PEOPLE object
        try:
            # Load payments
            tx_df = pd.read_csv(CSV_FILE, keep_default_na=False)
            # Normalize
            tx_df['date'] = pd.to_datetime(tx_df.get('date'), errors='coerce')
            tx_df['person'] = tx_df.get('person').astype(str)
            tx_df['amount'] = pd.to_numeric(tx_df.get('amount'), errors='coerce').fillna(0.0)
            tx_df['type'] = tx_df.get('type').astype(str).str.strip().str.lower()
            tx_df['payment_method'] = tx_df.get('payment_method').astype(str).str.strip().str.lower()
            tx_df['cheque_status'] = tx_df.get('cheque_status').astype(str)
            # prefer transaction_status, fallback to status
            status_col = 'transaction_status' if 'transaction_status' in tx_df.columns else ('status' if 'status' in tx_df.columns else None)
            if status_col:
                tx_df[status_col] = tx_df[status_col].astype(str).str.strip().str.lower()
            tx_df['reference_number'] = tx_df.get('reference_number').astype(str)
            tx_df['description'] = tx_df.get('description').astype(str)

            # Build rows
            tx_rows = []
            tx_df_sorted = tx_df.sort_values('date')
            for row in tx_df_sorted.itertuples(index=False):
                d = getattr(row, 'date')
                date_str = d.strftime('%Y-%m-%d') if pd.notna(d) else ''
                person = getattr(row, 'person')
                amount = float(getattr(row, 'amount') or 0.0)
                ttype = (getattr(row, 'type') or '').lower()
                method = (getattr(row, 'payment_method') or '').lower()
                cheque_status = getattr(row, 'cheque_status') or ''
                ref = getattr(row, 'reference_number') or ''
                desc = getattr(row, 'description') or ''
                status_val = getattr(row, status_col) if status_col else ''

                type_label = 'Paid' if ttype == 'i_paid' else ('Received' if ttype == 'paid_to_me' else ttype.title())
                cheque_disp = '-' if not cheque_status or cheque_status in ('nan', 'none', 'null') else cheque_status.title()
                status_disp = '-' if not status_val or status_val in ('nan', 'none', 'null') else status_val.title()

                tx_rows.append(
                    f"<tr data-date=\"{date_str}\" data-person=\"{person.lower()}\" data-type=\"{ttype}\" data-method=\"{method}\" data-cheque-status=\"{cheque_status.lower()}\" data-reference-number=\"{ref}\" data-amount-raw=\"{amount}\">\n"
                    f"    <td data-label=\"Date\">{date_str}</td>\n"
                    f"    <td data-label=\"Person\">{person}</td>\n"
                    f"    <td data-label=\"Amount\">Rs. {amount:,.2f}</td>\n"
                    f"    <td data-label=\"Type\">{type_label}</td>\n"
                    f"    <td data-label=\"Method\">{method.title() if method else '-'}</td>\n"
                    f"    <td data-label=\"Cheque Status\"><span class=\"status \">{cheque_disp}</span></td>\n"
                    f"    <td data-label=\"Reference No.\">{ref}</td>\n"
                    f"    <td data-label=\"Status\"><span class=\"status {status_val}\">{status_disp}</span></td>\n"
                    f"    <td data-label=\"Description\">{desc}</td>\n"
                    f"</tr>"
                )
            tx_html = "".join(tx_rows)

            # Replace transactions tbody
            updated = re.sub(
                r"(<table id=\"transactions-table\">[\s\S]*?<tbody>)([\s\S]*?)(</tbody>)",
                lambda m: m.group(1) + tx_html + m.group(3),
                updated,
                flags=re.DOTALL
            )

            # PEOPLE object from people.csv categories
            ppl = pd.read_csv(PEOPLE_FILE, keep_default_na=False)
            ppl['name'] = ppl.get('name').astype(str).str.strip()
            ppl['category'] = ppl.get('category').astype(str).str.strip().str.lower()
            investors = [n for n, c in zip(ppl['name'], ppl['category']) if c == 'investor']
            clients = [n for n, c in zip(ppl['name'], ppl['category']) if c == 'client']
            allp = sorted(set(ppl['name']))

            def _arr(vals):
                return '[' + ','.join([f'"{v}"' for v in vals]) + ']'

            people_block = (
                "const PEOPLE = {" +
                f"investor: {_arr(investors)}," +
                f" client: {_arr(clients)}," +
                f" all: {_arr(allp)}" +
                "};"
            )

            updated = re.sub(
                r"const PEOPLE = \{[\s\S]*?\};",
                people_block,
                updated,
                flags=re.DOTALL
            )
        except Exception:
            pass

        if updated != html:
            _write_text(SUMMARY_FILE, updated)
            return True
        return False
    except Exception:
        return False

def update_public_html_if_stale() -> None:
    """Regenerate docs HTML and push on every run. Safe: only writes if content changes."""
    try:
        regenerate_docs_index_html()
        try:
            git_push()
        except Exception:
            pass
    except Exception:
        pass

# Data Processing Classes


class DataCleaner:
    @staticmethod
    def clean_payments_data(data_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize payments data"""
        if data_df.empty:
            return data_df

        df = data_df.copy()

        # Ensure all expected columns exist and are of string type
        for col in Config.PAYMENT_COLUMNS:
            if col not in df.columns:
                df[col] = ''
            df[col] = df[col].astype(str).replace(
                'nan', '').replace('None', '').str.strip()

        # Convert numeric and date fields
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Process cash payments
        cash_mask = df['payment_method'].str.lower() == 'cash'
        df.loc[cash_mask, 'cheque_status'] = ''

        # Clean each row
        df = df.apply(DataCleaner._clean_payment_row, axis=1)

        # Remove empty rows
        df = df[~((df['date'].isna()) & (
            df['person'] == '') & (df['amount'] == 0.0))]

        return df

    @staticmethod
    def _clean_payment_row(row: pd.Series) -> pd.Series:
        """Clean individual payment row data"""
        row = row.copy()
        ref_num = str(row['reference_number']).lower()
        trans_status = str(row['transaction_status']).lower()
        cheque_status = str(row['cheque_status']).lower()
        payment_method = str(row['payment_method']).lower()

        # Clean transaction status
        if ref_num in Config.VALID_TRANSACTION_STATUSES:
            if trans_status not in Config.VALID_TRANSACTION_STATUSES:
                row['transaction_status'] = ref_num
            row['reference_number'] = ''

        # Clean cheque status
        if ref_num in Config.VALID_CHEQUE_STATUSES and payment_method == 'cheque':
            if cheque_status not in Config.VALID_CHEQUE_STATUSES:
                row['cheque_status'] = ref_num
            row['reference_number'] = ''

        # Set default statuses
        if trans_status not in Config.VALID_TRANSACTION_STATUSES:
            row['transaction_status'] = 'completed'

        if payment_method == 'cheque':
            if cheque_status not in Config.VALID_CHEQUE_STATUSES:
                row['cheque_status'] = 'processing'
        else:
            row['cheque_status'] = ''

        return row


class ExpenseCleaner:
    @staticmethod
    def clean_client_expenses_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize client expenses data"""
        if df.empty:
            return df

        df = df.copy()

        # Ensure all expected columns exist and are of string type
        for col in Config.CLIENT_EXPENSE_COLUMNS:
            if col not in df.columns:
                df[col] = ''
            df[col] = df[col].astype(str).replace(
                'nan', '').replace('None', '').str.strip()

        # Convert numeric fields
        df['expense_amount'] = pd.to_numeric(
            df['expense_amount'], errors='coerce').fillna(0.0)
        df['expense_quantity'] = pd.to_numeric(
            df['expense_quantity'], errors='coerce').fillna(1.0)

        # Convert date field
        df['expense_date'] = pd.to_datetime(
            df['expense_date'], errors='coerce')

        # Remove rows that are entirely empty
        df = df[~((df['expense_date'].isna()) &
                  (df['expense_person'] == '') &
                  (df['expense_amount'] == 0.0) &
                  (df['expense_quantity'] == 0.0))]

        return df

# Function to initialize CSV files if they don't exist


def init_files():
    try:
        # Initialize payments.csv
        if not os.path.exists(CSV_FILE):
            pd.DataFrame(columns=[
                "date", "person", "amount", "type", "status",
                "description", "payment_method", "reference_number",
                "cheque_status", "transaction_status"
            ]).to_csv(CSV_FILE, index=False)
            st.toast("Created new " + str(CSV_FILE))
        else:
            # Load, clean, and save existing payments data
            df = pd.read_csv(CSV_FILE, dtype={
                             'reference_number': str}, keep_default_na=False)
            df['reference_number'] = df['reference_number'].apply(
                lambda x: '' if pd.isna(x) or str(x).strip().lower() == 'nan' or str(
                    x).strip().lower() == 'none' else str(x).strip()
            )
            # Handle migration from old receipt_number/cheque_number columns
            if 'receipt_number' in df.columns or 'cheque_number' in df.columns:
                df['reference_number'] = df.apply(
                    lambda row: row['receipt_number'] if row['payment_method'] == 'cash'
                    else (row['cheque_number'] if row['payment_method'] == 'cheque' else ''),
                    axis=1
                ).fillna('')
                df = df.drop(columns=['receipt_number',
                             'cheque_number'], errors='ignore')
                st.toast("Migrated old reference number columns.")
            df = DataCleaner.clean_payments_data(df)
            df.to_csv(CSV_FILE, index=False)
            st.toast("Payments data cleaned and saved.")

        # Initialize people.csv
        if not os.path.exists(PEOPLE_FILE):
            pd.DataFrame(columns=["name", "category"]).to_csv(
                PEOPLE_FILE, index=False)
            st.toast(f"Created new {PEOPLE_FILE}")
        else:
            # Load and ensure 'category' column exists for people
            df = pd.read_csv(PEOPLE_FILE)
            if 'category' not in df.columns:
                df['category'] = 'client'  # Default category
                df.to_csv(PEOPLE_FILE, index=False)
            df['name'] = df['name'].astype(str)
            df.to_csv(PEOPLE_FILE, index=False)

        # Initialize client_expenses.csv
        if not os.path.exists(CLIENT_EXPENSES_FILE):
            pd.DataFrame(columns=[
                "original_transaction_ref_num", "expense_date", "expense_person",
                "expense_category", "expense_amount", "expense_quantity", "expense_description"
            ]).to_csv(CLIENT_EXPENSES_FILE, index=False)
            st.toast(f"Created new {CLIENT_EXPENSES_FILE}")
        else:
            # Load, clean, and ensure 'expense_quantity' for client expenses
            df_exp = pd.read_csv(CLIENT_EXPENSES_FILE, dtype={
                                 'original_transaction_ref_num': str, 'expense_person': str}, keep_default_na=False)
            df_exp = ExpenseCleaner.clean_client_expenses_data(df_exp)
            if 'expense_quantity' not in df_exp.columns:
                # Default quantity to 1.0 for existing entries
                df_exp['expense_quantity'] = 1.0
                st.toast("Added 'expense_quantity' column to client_expenses.csv")
            df_exp.to_csv(CLIENT_EXPENSES_FILE, index=False)
            st.toast("Client expenses data cleaned and saved.")

    except pd.errors.EmptyDataError as e:
        st.error(f"Error reading CSV file: {e}")
        logger.error("Empty data file encountered: %s", str(e))
    except (OSError, IOError) as e:
        st.error(f"File system error: {e}")
        logger.error("File access error: %s", str(e))
    except ValueError as e:
        st.error(f"Data validation error: {e}")
        logger.error("Data validation failed: %s", str(e))


init_files()

# PDF Generation Helper Functions


class PDFHelpers:
    """A utility class for generating PDF reports with consistent styling and formatting."""

    @staticmethod
    def generate_header(pdf: FPDF, report_title: str, person_name: str, date_start: datetime, date_end: datetime) -> None:
        """
        Generate a professional header for PDF reports.

        Args:
            pdf: The FPDF instance to add the header to
            report_title: The title to display at the top of the report
            person_name: The name of the person the report is for
            date_start: The start date of the report period
            date_end: The end date of the report period
        """
        try:
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'PAYMENT TRACKER SYSTEM', 0, 1, 'C')
            pdf.ln(2)

            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, report_title, 0, 1, 'C')
            pdf.ln(5)

            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 8, f'Client/Person: {person_name}', 0, 1, 'L')
            pdf.cell(
                0, 8, f'Period: {date_start.strftime("%Y-%m-%d")} to {date_end.strftime("%Y-%m-%d")}', 0, 1, 'L')
            pdf.cell(
                0, 8, f'Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 0, 1, 'L')
            pdf.ln(5)

            # Add a line separator
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
        except (AttributeError, ValueError) as e:
            logger.error("Error in PDF header generation: %s", str(e))
            st.error("Failed to generate PDF header")
            raise

    @staticmethod
    def generate_footer(pdf: FPDF) -> None:
        """Generate a professional footer for PDF reports"""
        try:
            pdf.ln(10)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

            pdf.set_font('Arial', 'I', 8)
            pdf.cell(
                0, 5, f'Report generated by Payment Tracker System - Page {pdf.page_no()}', 0, 1, 'C')
            pdf.cell(
                0, 5, f'Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 0, 1, 'C')
        except (AttributeError, ValueError) as e:
            logger.error("Error in PDF footer generation: %s", str(e))
            st.error("Failed to generate PDF footer")
            raise


# Report Generation Functions

def _ensure_invoice_dir() -> str:
    os.makedirs(INVOICE_DIR, exist_ok=True)
    return INVOICE_DIR


def _pdf_table_header(pdf: FPDF) -> None:
    pdf.set_font('Arial', 'B', 10)
    headers = ["Date", "Debit", "Credit", "Description", "Serial ID", "Receipt ID"]
    widths = [25, 25, 25, 70, 22, 22]
    for h, w in zip(headers, widths):
        pdf.cell(w, 8, h, 1, 0, 'C')
    pdf.ln(8)


def _pdf_table_row(pdf: FPDF, date_str: str, debit: float, credit: float, desc: str, serial_id: str, receipt_id: str) -> None:
    pdf.set_font('Arial', '', 9)
    widths = [25, 25, 25, 70, 22, 22]
    cells = [
        date_str,
        f"{debit:,.2f}" if debit else "",
        f"{credit:,.2f}" if credit else "",
        str(desc)[:120],
        str(serial_id),
        str(receipt_id)
    ]
    for v, w in zip(cells, widths):
        pdf.cell(w, 7, v, 1, 0, 'L')
    pdf.ln(7)


def generate_inquiry_pdf(person_name: str, start_date: datetime, end_date: datetime) -> Optional[str]:
    try:
        df = pd.read_csv(CSV_FILE, dtype={'reference_number': str}, keep_default_na=False)
        df['reference_number'] = df['reference_number'].astype(str).str.strip()
        df = clean_payments_data(df)
        mask = (
            (df['person'].astype(str).str.strip() == str(person_name).strip()) &
            (df['type'].str.lower() == 'i_paid') &
            (df['date'] >= pd.to_datetime(start_date)) &
            (df['date'] <= pd.to_datetime(end_date))
        )
        sub = df.loc[mask].copy().sort_values('date')

        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        PDFHelpers.generate_header(pdf, 'Inquiry Report', person_name, start_date, end_date)
        _pdf_table_header(pdf)

        total_debit = 0.0
        total_credit = 0.0
        for i, row in enumerate(sub.itertuples(index=False), start=1):
            date_str = getattr(row, 'date').strftime('%Y-%m-%d') if pd.notna(getattr(row, 'date')) else ''
            debit = float(getattr(row, 'amount') or 0.0)
            _pdf_table_row(
                pdf,
                date_str,
                debit,
                0.0,
                getattr(row, 'description'),
                str(i),
                getattr(row, 'reference_number')
            )
            total_debit += debit

        # Totals
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(25, 8, 'Totals', 1)
        pdf.cell(25, 8, f"{total_debit:,.2f}", 1, 0, 'R')
        pdf.cell(25, 8, f"{total_credit:,.2f}", 1, 0, 'R')
        pdf.cell(114, 8, '', 1, 1)

        net_balance = total_debit - total_credit
        pdf.ln(4)
        pdf.cell(0, 8, f'Net Balance (Debit - Credit): {net_balance:,.2f}', 0, 1, 'R')
        PDFHelpers.generate_footer(pdf)

        # Ensure person-specific report directory: reports/<person_name_sanitized>/
        safe_person = "".join(c for c in str(person_name) if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        out_dir = os.path.join('reports', safe_person)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'inquiry_{safe_person}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.pdf')
        pdf.output(out_path)
        return out_path
    except Exception as e:
        st.error(f"Error generating inquiry PDF: {e}")
        logger.exception(e)
        return None


def generate_bill_pdf(person_name: str, start_date: datetime, end_date: datetime, category: Optional[str] = None) -> Optional[str]:
    try:
        df = pd.read_csv(CLIENT_EXPENSES_FILE, dtype={'original_transaction_ref_num': str}, keep_default_na=False)
        df = clean_client_expenses_data(df)
        mask = (
            (df['expense_person'].astype(str).str.strip() == str(person_name).strip()) &
            (df['expense_date'] >= pd.to_datetime(start_date)) &
            (df['expense_date'] <= pd.to_datetime(end_date))
        )
        # Add category filter if specified
        if category and category != "All":
            mask = mask & (df['expense_category'].astype(str).str.strip() == str(category).strip())
        sub = df.loc[mask].copy().sort_values('expense_date')
        sub['line_total'] = sub['expense_amount'] * sub['expense_quantity']

        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        PDFHelpers.generate_header(pdf, 'Bill Report', person_name, start_date, end_date)
        _pdf_table_header(pdf)

        total_debit = 0.0
        total_credit = 0.0
        for i, row in enumerate(sub.itertuples(index=False), start=1):
            date_val = getattr(row, 'expense_date')
            date_str = date_val.strftime('%Y-%m-%d') if pd.notna(date_val) else ''
            credit = float(getattr(row, 'line_total') or 0.0)
            desc = getattr(row, 'expense_description')
            receipt_id = getattr(row, 'original_transaction_ref_num')
            _pdf_table_row(pdf, date_str, 0.0, credit, desc, str(i), receipt_id)
            total_credit += credit

        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(25, 8, 'Totals', 1)
        pdf.cell(25, 8, f"{total_debit:,.2f}", 1, 0, 'R')
        pdf.cell(25, 8, f"{total_credit:,.2f}", 1, 0, 'R')
        pdf.cell(114, 8, '', 1, 1)

        net_balance = total_debit - total_credit
        pdf.ln(4)
        pdf.cell(0, 8, f'Net Balance (Debit - Credit): {net_balance:,.2f}', 0, 1, 'R')
        PDFHelpers.generate_footer(pdf)

        # Ensure person-specific report directory: reports/<person_name_sanitized>/
        safe_person = "".join(c for c in str(person_name) if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        out_dir = os.path.join('reports', safe_person)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'bill_{safe_person}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.pdf')
        pdf.output(out_path)
        return out_path
    except Exception as e:
        st.error(f"Error generating bill PDF: {e}")
        logger.exception(e)
        return None


def generate_invoice_pdf(person_name: str, start_date: datetime, end_date: datetime, category: Optional[str] = None) -> Optional[str]:
    try:
        pay = pd.read_csv(CSV_FILE, dtype={'reference_number': str}, keep_default_na=False)
        pay['reference_number'] = pay['reference_number'].astype(str).str.strip()
        pay = clean_payments_data(pay)
        pay_mask = (
            (pay['person'].astype(str).str.strip() == str(person_name).strip()) &
            (pay['type'].str.lower() == 'i_paid') &
            (pay['date'] >= pd.to_datetime(start_date)) &
            (pay['date'] <= pd.to_datetime(end_date))
        )
        pay_sub = pay.loc[pay_mask, ['date', 'amount', 'description', 'reference_number']].copy()
        pay_sub['kind'] = 'debit'

        exp = pd.read_csv(CLIENT_EXPENSES_FILE, dtype={'original_transaction_ref_num': str}, keep_default_na=False)
        exp = clean_client_expenses_data(exp)
        exp_mask = (
            (exp['expense_person'].astype(str).str.strip() == str(person_name).strip()) &
            (exp['expense_date'] >= pd.to_datetime(start_date)) &
            (exp['expense_date'] <= pd.to_datetime(end_date))
        )
        # Add category filter if specified
        if category and category != "All":
            exp_mask = exp_mask & (exp['expense_category'].astype(str).str.strip() == str(category).strip())
        exp_sub = exp.loc[exp_mask, ['expense_date', 'expense_amount', 'expense_quantity', 'expense_description', 'original_transaction_ref_num']].copy()
        exp_sub['line_total'] = exp_sub['expense_amount'] * exp_sub['expense_quantity']
        exp_sub['kind'] = 'credit'

        # Normalize columns
        pay_sub = pay_sub.rename(columns={
            'date': 'date', 'amount': 'debit', 'description': 'description', 'reference_number': 'receipt_id'
        })
        pay_sub['credit'] = 0.0

        exp_sub = exp_sub.rename(columns={
            'expense_date': 'date', 'line_total': 'credit', 'expense_description': 'description', 'original_transaction_ref_num': 'receipt_id'
        })
        exp_sub['debit'] = 0.0

        # Combine
        cols = ['date', 'debit', 'credit', 'description', 'receipt_id']
        combined = pd.concat([
            pay_sub[cols],
            exp_sub[cols]
        ], ignore_index=True)
        combined = combined.sort_values('date')

        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        PDFHelpers.generate_header(pdf, 'Invoice (Combined)', person_name, start_date, end_date)
        _pdf_table_header(pdf)

        total_debit = 0.0
        total_credit = 0.0
        for i, row in enumerate(combined.itertuples(index=False), start=1):
            date_val = getattr(row, 'date')
            date_str = date_val.strftime('%Y-%m-%d') if pd.notna(date_val) else ''
            debit = float(getattr(row, 'debit') or 0.0)
            credit = float(getattr(row, 'credit') or 0.0)
            desc = getattr(row, 'description')
            receipt_id = getattr(row, 'receipt_id')
            _pdf_table_row(pdf, date_str, debit, credit, desc, str(i), receipt_id)
            total_debit += debit
            total_credit += credit

        pdf.set_font('Arial', 'B', 10)
        pdf.cell(25, 8, 'Totals', 1)
        pdf.cell(25, 8, f"{total_debit:,.2f}", 1, 0, 'R')
        pdf.cell(25, 8, f"{total_credit:,.2f}", 1, 0, 'R')
        pdf.cell(114, 8, '', 1, 1)

        net_balance = total_debit - total_credit
        pdf.ln(4)
        pdf.cell(0, 8, f'Net Balance (Debit - Credit): {net_balance:,.2f}', 0, 1, 'R')
        PDFHelpers.generate_footer(pdf)

        # Ensure person-specific report directory: reports/<person_name_sanitized>/
        safe_person = "".join(c for c in str(person_name) if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        out_dir = os.path.join('reports', safe_person)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'invoice_{safe_person}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.pdf')
        pdf.output(out_path)
        return out_path
    except Exception as e:
        st.error(f"Error generating invoice PDF: {e}")
        logger.exception(e)
        return None

# Function to push changes to GitHub (for public summary)


def git_push():
    try:
        repo = Repo(REPO_PATH)
        if repo.is_dirty(untracked_files=True):
            repo.git.add(update=True)
            repo.git.add(all=True)
        if repo.index.diff("HEAD"):
            repo.index.commit("Automated update: payment records")
        else:
            return True  # No changes to commit
        origin = repo.remote(name='origin')
        origin.push()
        st.success("GitHub updated successfully!")
        return True
    except git.exc.InvalidGitRepositoryError:
        st.error("Invalid Git repository")
        st.warning("Please ensure this is a valid Git repository")
        return False
    except git.exc.GitCommandError as e:
        st.error(f"Git command error: {str(e)}")
        st.warning("Please check your Git configuration and permissions")
        return False
    except (FileNotFoundError, PermissionError) as e:
        st.error(f"File system error: {str(e)}")
        st.warning("Please check file permissions and repository path")
        return False
    except Exception as e:
        st.error(f"Unexpected error while updating GitHub: {str(e)}")
        st.warning(
            "If the issue persists, try running 'git push' manually in your terminal for detailed errors.")
        return False

# Function to create backup of all data files


def create_backup():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"payments_backup_{timestamp}.zip"
        backup_path = os.path.join(os.getcwd(), "backups")

        # Create backups directory if it doesn't exist
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)

        backup_file_path = os.path.join(backup_path, backup_filename)

        # Create zip file with all data files
        with zipfile.ZipFile(backup_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            data_files = [CSV_FILE, CLIENT_EXPENSES_FILE, PEOPLE_FILE]
            success_count = 0
            for file_path in data_files:
                if os.path.exists(file_path):
                    # Get the filename without the full path
                    arcname = os.path.basename(file_path)
                    # Add file to the zip archive
                    zipf.write(file_path, arcname)
                    st.toast(f"Added {arcname} to backup")
                    success_count += 1

        if success_count > 0:
            st.success(f"Backup created successfully at {backup_file_path}")
            return backup_file_path
        else:
            st.warning("No files were found to backup")
            return None
    except Exception as e:
        st.error(f"Error creating backup: {e}")
        return None

# Function to restore data from backup


def restore_backup(uploaded_file):
    try:
        # Create temporary directory for extraction
        temp_dir = os.path.join(os.getcwd(), "temp_restore")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save uploaded file temporarily
        temp_zip_path = os.path.join(temp_dir, "backup.zip")
        with open(temp_zip_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Extract backup files
        with zipfile.ZipFile(temp_zip_path, 'r') as zipf:
            zipf.extractall(temp_dir)

        # Restore files to their original locations
        restored_files = []
        data_files = {
            "payments.csv": CSV_FILE,
            "client_expenses.csv": CLIENT_EXPENSES_FILE,
            "people.csv": PEOPLE_FILE
        }

        for backup_name, original_path in data_files.items():
            backup_file_path = os.path.join(temp_dir, backup_name)
            if os.path.exists(backup_file_path):
                shutil.copy2(backup_file_path, original_path)
                restored_files.append(backup_name)

        # Clean up temporary files
        shutil.rmtree(temp_dir)

        # After restore, regenerate summary and auto-commit/push
        try:
            if os.path.exists(CSV_FILE):
                payments_df = pd.read_csv(CSV_FILE, dtype={'reference_number': str}, keep_default_na=False)
                payments_df = DataCleaner.clean_payments_data(payments_df)
                generate_html_summary(payments_df)
            changed = [CSV_FILE, CLIENT_EXPENSES_FILE, PEOPLE_FILE]
            changed = [p for p in changed if os.path.exists(p)] + ([SUMMARY_FILE] if os.path.exists(SUMMARY_FILE) else [])
            commit_and_push_files(changed, "Restore backup and regenerate summary")
        except Exception as e:
            st.warning(f"Restore completed, but post-restore summary or push failed: {e}")

        return restored_files
    except zipfile.BadZipFile as e:
        st.error("Invalid backup file format")
        logger.error("Bad zip file: %s", str(e))
        return []
    except (OSError, IOError) as e:
        st.error(f"File system error while restoring backup: {e}")
        logger.error("File access error during restore: %s", str(e))
        return []
    except shutil.Error as e:
        st.error(f"Error copying files during restore: {e}")
        logger.error("File copy error: %s", str(e))
        return []

# Function to prepare DataFrame for display in HTML summary


def prepare_dataframe_for_display(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for display in HTML by cleaning and formatting values.

    Args:
        input_df: Input DataFrame with raw data

    Returns:
        DataFrame with cleaned and formatted values for display
    """
    df_display = input_df.copy()
    # Ensure all relevant columns are strings and handle missing values
    for col in ['reference_number', 'cheque_status', 'transaction_status', 'payment_method', 'description', 'person', 'type', 'status']:
        if col in df_display.columns:
            df_display[col] = df_display[col].astype(str).replace(
                'nan', '').replace('None', '').str.strip()
        else:
            df_display[col] = ''  # Add missing column if not present

    df_display['amount'] = pd.to_numeric(
        df_display['amount'], errors='coerce').fillna(0.0)
    df_display['amount_display'] = df_display['amount'].apply(
        lambda x: f"Rs. {x:,.2f}")

    df_display['date_parsed'] = pd.to_datetime(df_display['date'], errors='coerce')
    df_display['formatted_date'] = df_display['date_parsed'].dt.strftime(
        '%Y-%m-%d').fillna('-')

    # Clean and display cheque status
    df_display['cheque_status_cleaned'] = df_display.apply(
        lambda row: None if row['payment_method'].lower() == 'cash' else (
            str(row['cheque_status']) if pd.notna(
                row['cheque_status']) else None
        ), axis=1
    )
    df_display['cheque_status_display'] = df_display['cheque_status_cleaned'].apply(
        lambda x: next((s.capitalize(
        ) for s in Config.VALID_CHEQUE_STATUSES if x is not None and str(x).lower() == s), '-')
    )
    # Clean and display transaction status
    df_display['transaction_status_display'] = df_display.apply(
        lambda row: str(row['transaction_status']).capitalize() if str(
            row['transaction_status']).lower() in [s.lower() for s in Config.VALID_TRANSACTION_STATUSES] else '-',
        axis=1
    )
    # Clean and display reference number
    df_display['reference_number_display'] = df_display.apply(
        lambda row: str(row['reference_number']) if str(
            row['reference_number']).strip() != '' else '-',
        axis=1
    )
    # Map transaction type for display
    df_display['type_display'] = df_display['type'].map(
        {'paid_to_me': 'Received', 'i_paid': 'Paid'}).fillna('')

    # Ensure payment_method is also explicitly handled for display
    df_display['payment_method_display'] = df_display['payment_method'].apply(
        lambda x: str(x).capitalize() if str(x).strip() != '' else '-'
    )

    return df_display

# Function to generate the public HTML summary


class HTMLGenerator:
    """Helper class for generating HTML summary components"""

    @staticmethod
    def calculate_totals(input_df: pd.DataFrame) -> dict:
        """Calculate summary totals from payment data"""
        df = input_df.copy()
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        df['type'] = df['type'].astype(str).str.lower()
        df['payment_method'] = df['payment_method'].astype(str).str.lower()
        df['transaction_status'] = df['transaction_status'].astype(
            str).str.lower()

        # Calculate payment totals by type and method
        payment_totals = df.groupby(['type', 'payment_method'])[
            'amount'].sum().unstack().fillna(0)

        return {
            'total_received': df[df['type'] == 'paid_to_me']['amount'].sum(),
            'pending_received': df[(df['type'] == 'paid_to_me') &
                                   (df['transaction_status'] == 'pending')]['amount'].sum(),
            'total_paid': df[df['type'] == 'i_paid']['amount'].sum(),
            'pending_paid': df[(df['type'] == 'i_paid') &
                               (df['transaction_status'] == 'pending')]['amount'].sum(),
            'cash_received': payment_totals.loc['paid_to_me', 'cash']
            if 'cash' in payment_totals.columns and 'paid_to_me' in payment_totals.index else 0,
            'cheque_received': payment_totals.loc['paid_to_me', 'cheque']
            if 'cheque' in payment_totals.columns and 'paid_to_me' in payment_totals.index else 0,
            'cash_paid': payment_totals.loc['i_paid', 'cash']
            if 'cash' in payment_totals.columns and 'i_paid' in payment_totals.index else 0,
            'cheque_paid': payment_totals.loc['i_paid', 'cheque']
            if 'cheque' in payment_totals.columns and 'i_paid' in payment_totals.index else 0,
            'net_balance': (df[df['type'] == 'i_paid']['amount'].sum() -
                            df[df['type'] == 'paid_to_me']['amount'].sum())
        }


def generate_html_summary(input_df: pd.DataFrame) -> None:
    """Generate HTML summary of payment data and save to file"""
    logger.info("Starting HTML summary generation")
    try:
        transactions_display = prepare_dataframe_for_display(input_df)
        totals = HTMLGenerator.calculate_totals(input_df)
        # Use totals already calculated by HTMLGenerator
        df_for_totals = input_df.copy()

        # Load people data for filters
        people_df = pd.read_csv(PEOPLE_FILE)
        person_options_html = ''.join(
            f'<option value="{name}">{name}</option>' for name in sorted(people_df['name'].unique()))

        # Prepare categorized people lists for client/investor filtering in HTML summary
        if 'category' in people_df.columns:
            _cat = people_df['category'].astype(str).str.lower()
            investor_names = sorted(people_df[_cat == 'investor']['name'].dropna().astype(str).unique().tolist())
            client_names = sorted(people_df[_cat == 'client']['name'].dropna().astype(str).unique().tolist())
        else:
            investor_names = []
            client_names = []
        all_person_names = sorted(people_df['name'].dropna().astype(str).unique().tolist())

        # Turn lists into JS array literals (avoid needing json import)
        def _to_js_array(names: List[str]) -> str:
            return '[' + ','.join(f'"{n}"' for n in names) + ']'
        investor_js = _to_js_array(investor_names)
        client_js = _to_js_array(client_names)
        all_js = _to_js_array(all_person_names)

        # Load and process client expenses data
        client_expenses_df_all = pd.DataFrame()
        if os.path.exists(CLIENT_EXPENSES_FILE) and os.path.getsize(CLIENT_EXPENSES_FILE) > 0:
            client_expenses_df_all = pd.read_csv(CLIENT_EXPENSES_FILE, dtype={
                                                 'original_transaction_ref_num': str, 'expense_person': str}, keep_default_na=False)
            client_expenses_df_all = ExpenseCleaner.clean_client_expenses_data(
                client_expenses_df_all)
            client_expenses_df_all['total_line_amount'] = client_expenses_df_all['expense_amount'] * \
                client_expenses_df_all['expense_quantity']

        # Calculate total paid to clients (from 'I Paid' transactions)
        total_paid_to_clients = pd.DataFrame(
            columns=['client_name', 'total_paid_to_client'])
        if not df_for_totals[df_for_totals['type'] == 'i_paid'].empty:
            total_paid_to_clients = df_for_totals[df_for_totals['type'] == 'i_paid'].groupby(
                'person')['amount'].sum().reset_index()
            total_paid_to_clients.rename(
                columns={'person': 'client_name', 'amount': 'total_paid_to_client'}, inplace=True)

        # Calculate total spent by clients (from client expenses)
        total_spent_by_clients = pd.DataFrame(
            columns=['client_name', 'total_spent_by_client'])
        if not client_expenses_df_all.empty:
            total_spent_by_clients = client_expenses_df_all.groupby(
                'expense_person')['total_line_amount'].sum().reset_index()
            total_spent_by_clients.rename(columns={
                                          'expense_person': 'client_name', 'total_line_amount': 'total_spent_by_client'}, inplace=True)

        # Merge and calculate remaining balance for client overview
        expected_client_summary_cols = [
            'client_name', 'total_paid_to_client', 'total_spent_by_client']
        summary_by_client_df = pd.merge(
            total_paid_to_clients,
            total_spent_by_clients,
            on='client_name',
            how='outer'
        )
        for col in expected_client_summary_cols:
            if col not in summary_by_client_df.columns:
                summary_by_client_df[col] = 0
        summary_by_client_df.fillna(0, inplace=True)
        summary_by_client_df['client_name'] = summary_by_client_df['client_name'].astype(
            str).fillna('')
        summary_by_client_df['total_paid_to_client'] = pd.to_numeric(
            summary_by_client_df['total_paid_to_client'], errors='coerce').fillna(0)
        summary_by_client_df['total_spent_by_client'] = pd.to_numeric(
            summary_by_client_df['total_spent_by_client'], errors='coerce').fillna(0)
        summary_by_client_df['remaining_balance'] = summary_by_client_df['total_paid_to_client'] - \
            summary_by_client_df['total_spent_by_client']

        # Generate HTML for client overview section
        client_overview_html = ""
        if not summary_by_client_df.empty:
            client_overview_html += """
            <h3 class="section-subtitle"><i class="fas fa-chart-pie"></i> Spending Overview by Client</h3>
            <table class="client-summary-table">
                <thead>
                    <tr>
                        <th>Client Name</th>
                        <th>Total Paid to Client</th>
                        <th>Total Spent by Client</th>
                        <th>Remaining Balance</th>
                    </tr>
                </thead>
                <tbody>
            """
            for idx, row in summary_by_client_df.iterrows():
                balance_class = 'positive-balance' if row['remaining_balance'] >= 0 else 'negative-balance'
                client_overview_html += f"""
                    <tr>
                        <td>{row['client_name']}</td>
                        <td>Rs. {row['total_paid_to_client']:,.2f}</td>
                        <td>Rs. {row['total_spent_by_client']:,.2f}</td>
                        <td class="{balance_class}">Rs. {row['remaining_balance']:,.2f}</td>
                    </tr>
                """
            client_overview_html += """
                </tbody>
            </table>
            """
        else:
            client_overview_html = "<p class='no-results'>No client spending overview available yet.</p>"

        # Generate HTML for detailed client expenses section
        detailed_expenses_html = ""
        total_client_expenses_grand = 0.0
        if not client_expenses_df_all.empty:
            total_client_expenses_grand = client_expenses_df_all['total_line_amount'].sum(
            )
            client_expenses_df_all = client_expenses_df_all.sort_values(
                'expense_date', ascending=False)

            detailed_expenses_html += f"""
            <h3 class="section-subtitle"><i class="fas fa-list-alt"></i> Detailed Client Expenses</h3>
            <div style="text-align: right; margin-bottom: 10px; font-weight: bold; font-size: 1.2em; color: #34495e;">
                Total Client Expenses: Rs. {total_client_expenses_grand:,.2f}
            </div>
            <table class="detailed-expenses-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Client Name</th>
                        <th>Category</th>
                        <th>Amount (Unit)</th>
                        <th>Quantity</th>
                        <th>Total Line Amount</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
            """
            for _, row in client_expenses_df_all.iterrows():
                _dt = pd.to_datetime(row['expense_date'], errors='coerce')
                expense_date_str = _dt.strftime('%Y-%m-%d') if pd.notna(_dt) else '-'
                expense_desc = row['expense_description'] if pd.notna(row['expense_description']) and str(row['expense_description']).strip() else '-'
                detailed_expenses_html += f"""
                    <tr>
                        <td>{expense_date_str}</td>
                        <td>{row['expense_person']}</td>
                        <td>{row['expense_category']}</td>
                        <td>Rs. {row['expense_amount']:,.2f}</td>
                        <td>{row['expense_quantity']:,.0f}</td>
                        <td>Rs. {row['total_line_amount']:,.2f}</td>
                        <td>{expense_desc}</td>
                    </tr>
                """
            detailed_expenses_html += """
                </tbody>
            </table>
            """
        else:
            detailed_expenses_html = "<p class='no-results'>No detailed client expenses to display yet.</p>"

        # Full HTML structure for the summary page
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Summary | Financial Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {{
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f7f6;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        }}
        header {{
            text-align: center;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
            margin-bottom: 30px;
        }}
        .logo {{
            font-size: 2.5em;
            font-weight: 700;
            color: #2c3e50;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 10px;
        }}
        .logo i {{
            color: #28a745;
            margin-right: 10px;
            font-size: 0.9em;
        }}
        .report-title {{
            font-size: 1.8em;
            color: #34495e;
            margin-bottom: 5px;
        }}
        .report-date {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        .report-date i {{
            margin-right: 5px;
            color: #95a5a6;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        .card {{
            background-color: #fdfdfd;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-left: 5px solid;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }}
        .card.received {{ border-left-color: #28a745; }}
        .card.paid {{ border-left-color: #dc3545; }}
        .card.balance {{ border-left-color: #007bff; }}

        .card-header {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }}
        .card-icon {{
            font-size: 1.8em;
            margin-right: 15px;
            padding: 12px;
            border-radius: 50%;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .received .card-icon {{ background-color: #28a745; }}
        .paid .card-icon {{ background-color: #dc3545; }}
        .balance .card-icon {{ background-color: #007bff; }}

        .card-title {{
            font-size: 1.1em;
            color: #555;
            margin-bottom: 3px;
        }}
        .card-amount {{
            font-size: 1.9em;
            font-weight: 600;
            color: #2c3e50;
        }}
        .card-details {{
            font-size: 0.9em;
            color: #666;
            padding-left: 55px;
        }}
        .card-details div {{
            margin-bottom: 5px;
        }}
        .card-details i {{
            margin-right: 8px;
            color: #999;
        }}

        .section-title {{
            font-size: 1.6em;
            color: #34495e;
            margin-bottom: 25px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            display: flex;
            align-items: center;
        }}
        .section-title i {{
            margin-right: 10px;
            color: #007bff;
        }}
        .section-subtitle {{
            font-size: 1.3em;
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
            display: flex;
            align-items: center;
        }}
        .section-subtitle i {{
            margin-right: 8px;
            color: #28a745;
        }}


        .filters {{
            background-color: #fdfdfd;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
            margin-bottom: 40px;
        }}
        .filter-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .filter-group label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #555;
        }}
        .filter-group select,
        .filter-group input[type="date"],
        .filter-group input[type="text"] {{
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-family: 'Poppins', sans-serif;
            font-size: 0.95em;
            color: #333;
            box-sizing: border-box;
            background-color: #fff;
        }}
        .filter-group select:focus,
        .filter-group input[type="date"]:focus,
        .filter-group input[type="text"]:focus {{
            border-color: #007bff;
            box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.15);
            outline: none;
        }}
        .filter-actions {{
            text-align: right;
            margin-top: 20px;
        }}
        .filter-btn {{
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            transition: background-color 0.3s ease, transform 0.2s ease;
            margin-left: 10px;
            display: inline-flex;
            align-items: center;
        }}
        .filter-btn i {{
            margin-right: 8px;
        }}
        .filter-btn:hover {{
            background-color: #0056b3;
            transform: translateY(-2px);
        }}
        .filter-btn.reset-btn {{
            background-color: #6c757d;
        }}
        .filter-btn.reset-btn:hover {{
            background-color: #5a6268;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 40px;
            background-color: #fdfdfd;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #e9ecef;
            color: #495057;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
        }}
        tbody tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tbody tr:hover {{
            background-color: #f1f1f1;
            transform: scale(1.005);
            transition: background-color 0.2s ease, transform 0.2s ease;
        }}
        .status {{
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: 500;
            font-size: 0.85em;
            color: #fff;
            display: inline-block;
        }}
        .status.completed {{ background-color: #28a745; }}
        .status.pending {{ background-color: #ffc107; color: #333; }}
        .status.received-given {{ background-color: #6c757d; }}
        .status.processing {{ background-color: #007bff; }}
        .status.bounced {{ background-color: #dc3545; }}
        .status.processing-done {{ background-color: #20c997; }}

        .no-results {{
            text-align: center;
            padding: 50px 20px;
            background-color: #fdfdfd;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
            margin-bottom: 40px;
            color: #7f8c8d;
        }}
        .no-results p {{
            font-size: 1.1em;
            margin-top: 10px;
        }}

        .footer {{
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
            font-size: 0.85em;
            margin-top: 30px;
        }}
        .footer p {{
            margin: 5px 0;
        }}
        .footer i {{
            margin-right: 5px;
            color: #95a5a6;
        }}

        .tabs {{
            margin-top: 30px;
            margin-bottom: 40px;
        }}
        .tab-buttons {{
            display: flex;
            border-bottom: 2px solid #ddd;
            margin-bottom: 20px;
        }}
        .tab-button {{
            padding: 12px 25px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 500;
            color: #555;
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 5px;
            transition: all 0.3s ease;
        }}
        .tab-button:hover {{
            background-color: #eee;
            color: #333;
        }}
        .tab-button.active {{
            background-color: #ffffff;
            color: #007bff;
            border-color: #007bff;
            border-bottom: 2px solid #ffffff;
            transform: translateY(2px);
            box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.05);
            z-index: 1;
        }}
        .tab-content {{
            background-color: #ffffff;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
            position: relative;
            top: -2px;
        }}
        .tab-pane {{
            display: none;
        }}
        .tab-pane.active {{
            display: block;
        }}
        .download-button-container {{
            text-align: right;
            margin-bottom: 20px;
        }}
        .download-button {{
            background-color: #17a2b8;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            transition: background-color 0.3s ease;
            display: inline-flex;
            align-items: center;
        }}
        .download-button i {{
            margin-right: 8px;
        }}
        .download-button:hover {{
            background-color: #117a8b;
        }}
        .download-button:disabled {{
            background-color: #6c757d;
            cursor: not-allowed;
        }}
        .balance-summary-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background-color: #eaf6ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 1.2em;
            font-weight: 600;
            color: #0056b3;
        }}
        .balance-summary-container.negative {{
            background-color: #fbecec;
            border-color: #f5c6cb;
            color: #721c24;
        }}
        .balance-label {{
            font-size: 0.9em;
            color: #6a6a6a;
        }}
        .balance-amount {{
            font-size: 1.5em;
            font-weight: 700;
            margin-left: 10px;
        }}
        .positive-balance {{ color: #28a745; }}
        .negative-balance {{ color: #dc3545; }}

        /* New styles for responsive tables */
        @media (max-width: 768px) {{
            .summary-cards {{ grid-template-columns: 1fr; }}
            .filter-grid {{ grid-template-columns: 1fr; }}
            .filter-actions {{ text-align: center; }}
            .filter-btn {{ width: 100%; margin-left: 0; margin-bottom: 10px; }}
            table, thead, tbody, th, td, tr {{ display: block; }}
            thead tr {{ position: absolute; top: -9999px; left: -9999px; }}
            tr {{ border: 1px solid #eee; margin-bottom: 15px; border-radius: 8px; overflow: hidden; }}
            td {{ border: none; position: relative; padding-left: 50%; text-align: right; }}
            td:before {{
                content: attr(data-label);
                position: absolute;
                left: 10px;
                width: 45%;
                padding-right: 10px;
                white-space: nowrap;
                text-align: left;
                font-weight: 600;
                color: #555;
            }}
            td:nth-of-type(1):before {{ content: "Date"; }}
            td:nth-of-type(2):before {{ content: "Person"; }}
            td:nth-of-type(3):before {{ content: "Amount"; }}
            td:nth-of-type(4):before {{ content: "Type"; }}
            td:nth-of-type(5):before {{ content: "Method"; }}
            td:nth-of-type(6):before {{ content: "Cheque Status"; }}
            td:nth-of-type(7):before {{ content: "Reference No."; }}
            td:nth-of-type(8):before {{ content: "Status"; }}
            td:nth-of-type(9):before {{ content: "Description"; }}
            .client-summary-table td:nth-of-type(1):before {{ content: "Client Name"; }}
            .client-summary-table td:nth-of-type(2):before {{ content: "Paid to Client"; }}
            .client-summary-table td:nth-of-type(3):before {{ content: "Spent by Client"; }}
            .client-summary-table td:nth-of-type(4):before {{ content: "Balance"; }}
            .detailed-expenses-table td:nth-of-type(1):before {{ content: "Date"; }}
            .detailed-expenses-table td:nth-of-type(2):before {{ content: "Client"; }}
            .detailed-expenses-table td:nth-of-type(3):before {{ content: "Category"; }}
            .detailed-expenses-table td:nth-of-type(4):before {{ content: "Amount (Unit)"; }}
            .detailed-expenses-table td:nth-of-type(5):before {{ content: "Quantity"; }}
            .detailed-expenses-table td:nth-of-type(6):before {{ content: "Total Amount"; }}
            .detailed-expenses-table td:nth-of-type(7):before {{ content: "Description"; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1 class="logo"><i class="fas fa-coins"></i>Payment Tracker</h1>
            <h2 class="report-title">Financial Summary Report</h2>
            <div class="report-date"><i class="far fa-calendar-alt"></i>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
        </header>

        <div class="summary-cards">
            <div class="card received">
                <div class="card-header">
                    <div class="card-icon"><i class="fas fa-arrow-alt-circle-down"></i></div>
                    <div>
                        <div class="card-title">Total Received</div>
                        <div class="card-amount">Rs. {totals['total_received']:,.2f}</div>
                    </div>
                </div>
                <div class="card-details">
                    <div><i class="fas fa-hand-holding-usd"></i> Cash: Rs. {totals['cash_received']:,.2f}</div>
                    <div><i class="fas fa-money-check"></i> Cheque: Rs. {totals['cheque_received']:,.2f}</div>
                    <div><i class="fas fa-hourglass-half"></i> Pending: Rs. {totals['pending_received']:,.2f}</div>
                </div>
            </div>
            <div class="card paid">
                <div class="card-header">
                    <div class="card-icon"><i class="fas fa-arrow-alt-circle-up"></i></div>
                    <div>
                        <div class="card-title">Total Paid</div>
                        <div class="card-amount">Rs. {totals['total_paid']:,.2f}</div>
                    </div>
                </div>
                <div class="card-details">
                    <div><i class="fas fa-hand-holding-usd"></i> Cash: Rs. {totals['cash_paid']:,.2f}</div>
                    <div><i class="fas fa-money-check"></i> Cheque: Rs. {totals['cheque_paid']:,.2f}</div>
                    <div><i class="fas fa-hourglass-half"></i> Pending: Rs. {totals['pending_paid']:,.2f}</div>
                </div>
            </div>
            <div class="card balance">
                <div class="card-header">
                    <div class="card-icon"><i class="fas fa-balance-scale"></i></div>
                    <div>
                        <div class="card-title">Net Balance</div>
                        <div class="card-amount">Rs.{totals['net_balance']:,.2f}</div>
                    </div>
                </div>
                <div class="card-details">
                    <div><i class="fas fa-info-circle"></i> Paid - Received (Debit - Credit)</div>
                    <div style="margin-top: 10px;">
                        {'<span style="color: #28a745;"><i class="fas fa-check-circle"></i> Positive Balance (Overpaid)</span>' if totals['net_balance'] >= 0 else '<span style="color: #dc3545;"><i class="fas fa-exclamation-circle"></i> Negative Balance (Owing)</span>'}
                    </div>
                </div>
            </div>
        </div>

        <div class="filters">
            <h2 class="section-title">
                <i class="fas fa-filter"></i> Filters
            </h2>
            <div class="filter-grid">
                <div class="filter-group">
                    <label for="start-date">Date Range</label>
                    <input type="date" id="start-date" class="date-filter">
                    <span style="display: inline-block; margin: 0 5px; font-size: 12px;">to</span>
                    <input type="date" id="end-date" class="date-filter">
                </div>
                <div class="filter-group">
                    <label for="name-filter">Person</label>
                    <select id="name-filter">
                        <option value="">All</option>
                        {person_options_html}
                    </select>
                </div>
                <div class="filter-group">
                    <label for="type-filter">Transaction Type</label>
                    <select id="type-filter">
                        <option value="">All</option>
                        <option value="paid_to_me">Received</option>
                        <option value="i_paid">Paid</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="method-filter">Payment Method</label>
                    <select id="method-filter">
                        <option value="">All</option>
                        <option value="cash">Cash</option>
                        <option value="cheque">Cheque</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="status-filter">Status</label>
                    <select id="status-filter">
                        <option value="">All</option>
                        <option value="completed">Completed</option>
                        <option value="pending">Pending</option>
                        <option value="received/given">Received/Given</option>
                        <option value="processing">Processing</option>
                        <option value="bounced">Bounced</option>
                        <option value="processing done">Processing Done</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="reference-number-filter">Reference Number</label>
                    <input type="text" id="reference-number-filter" placeholder="e.g., 12345">
                </div>
            </div>
            <div class="filter-actions">
                <button class="filter-btn reset-btn" onclick="resetFilters()"><i class="fas fa-undo-alt"></i> Reset</button>
                <button class="filter-btn" onclick="applyFilters()"><i class="fas fa-search"></i> Apply Filters</button>
            </div>
        </div>

        <div class="tabs">
            <div class="tab-buttons">
                <button class="tab-button active" data-tab="transactions-tab"><i class="fas fa-file-invoice"></i> All Transactions</button>
                <button class="tab-button" data-tab="client-expenses-tab"><i class="fas fa-money-check-alt"></i> Client Expenses</button>
            </div>
            <div class="tab-content">
                <div id="transactions-tab" class="tab-pane active">
                    <h2 class="section-title">
                        <i class="fas fa-exchange-alt"></i> All Transactions
                    </h2>
                    <div style="text-align: right; margin-bottom: 10px; font-weight: bold; font-size: 1.2em; color: #34495e;">
                        Total Displayed Amount: <span id="total-displayed-amount">Rs. 0.00</span>
                    </div>
                    <table id="transactions-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Person</th>
                                <th>Amount</th>
                                <th>Type</th>
                                <th>Method</th>
                                <th>Cheque Status</th>
                                <th>Reference No.</th>
                                <th>Status</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join([
            f"""<tr data-date="{transactions_display.loc[idx, 'formatted_date']}"
                                        data-person="{str(transactions_display.loc[idx, 'person']).lower()}"
                                        data-type="{str(transactions_display.loc[idx, 'type']).lower()}"
                                        data-method="{str(transactions_display.loc[idx, 'payment_method']).lower()}"
                                        data-cheque-status="{str(transactions_display.loc[idx, 'cheque_status']).lower()}"
                                        data-reference-number="{str(transactions_display.loc[idx, 'reference_number']).lower()}"
                                        data-amount-raw="{transactions_display.loc[idx, 'amount']}">
                                        <td data-label="Date">{transactions_display.loc[idx, 'formatted_date']}</td>
                                        <td data-label="Person">{transactions_display.loc[idx, 'person']}</td>
                                        <td data-label="Amount">{transactions_display.loc[idx, 'amount_display']}</td>
                                        <td data-label="Type">{transactions_display.loc[idx, 'type_display']}</td>
                                        <td data-label="Method">{transactions_display.loc[idx, 'payment_method_display']}</td>
                                        <td data-label="Cheque Status"><span class="status {'-'.join(str(transactions_display.loc[idx, 'cheque_status']).lower().split())}">{transactions_display.loc[idx, 'cheque_status_display']}</span></td>
                                        <td data-label="Reference No.">{transactions_display.loc[idx, 'reference_number_display']}</td>
                                        <td data-label="Status"><span class="status {str(transactions_display.loc[idx, 'transaction_status']).lower()}">{transactions_display.loc[idx, 'transaction_status_display']}</span></td>
                                        <td data-label="Description">{transactions_display.loc[idx, 'description'] if transactions_display.loc[idx, 'description'] else '-'}</td>
                                    </tr>"""
            for idx, row in transactions_display.iterrows()
        ]) if not transactions_display.empty else ''}
                        </tbody>
                    </table>
                </div>
                <div id="client-expenses-tab" class="tab-pane">
                    <h2 class="section-title">
                        <i class="fas fa-money-check-alt"></i> Client Expenses Overview
                    </h2>
                    {client_overview_html}
                    {detailed_expenses_html}
                </div>
            </div>
        </div>
    </div>
    <div class="footer">
        <p><i class="fas fa-file-alt"></i> This report was automatically generated by Payment Tracker System</p>
        <p><i class="far fa-copyright"></i> {datetime.now().year} All Rights Reserved</p>
    </div>
    <script>
        // People lists by category for dynamic person dropdown filtering
        const PEOPLE = {{
            investor: {investor_js},
            client: {client_js},
            all: {all_js}
        }};

        function populatePersonOptions() {{
            const typeVal = $('#type-filter').val(); // '' | 'paid_to_me' | 'i_paid'
            let list = PEOPLE.all;
            if (typeVal === 'paid_to_me') {{ // Received -> investors
                list = PEOPLE.investor;
            }} else if (typeVal === 'i_paid') {{ // Paid -> clients
                list = PEOPLE.client;
            }}
            const prev = $('#name-filter').val();
            const $sel = $('#name-filter');
            $sel.empty();
            $sel.append('<option value="">All</option>');
            list.forEach(function(name) {{
                $sel.append('<option value="' + name + '">' + name + '</option>');
            }});
            // Restore previous selection if still valid; otherwise default to All
            if (prev && list.map(String).includes(prev)) {{
                $sel.val(prev);
            }} else {{
                $sel.val('');
            }}
        }}

        $(document).ready(function() {{
            // Set default date filter to last month
            const today = new Date();
            const oneMonthAgo = new Date();
            oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
            $('#start-date').val(oneMonthAgo.toISOString().split('T')[0]);
            $('#end-date').val(today.toISOString().split('T')[0]);
            // Initialize person options based on current type selection
            populatePersonOptions();
            applyFilters();

            $('.tab-button').on('click', function() {{
                const tabId = $(this).data('tab');
                $('.tab-button').removeClass('active');
                $(this).addClass('active');
                $('.tab-pane').removeClass('active');
                $('#' + tabId).addClass('active');
            }});
            $('.tab-button[data-tab="transactions-tab"]').click();

            // Re-populate person list when type changes
            $('#type-filter').on('change', function() {{
                populatePersonOptions();
                applyFilters();
            }});
        }});

        function applyFilters() {{
            const startDate = $('#start-date').val();
            const endDate = $('#end-date').val();
            const person = $('#name-filter').val().toLowerCase();
            const type = $('#type-filter').val();
            const method = $('#method-filter').val().toLowerCase();
            const chequeStatus = $('#status-filter').val().toLowerCase();
            const referenceNumber = $('#reference-number-filter').val().toLowerCase();

            let visibleRows = 0;
            let totalAmount = 0;

            $('#transactions-table tbody tr').each(function() {{
                const rowDate = $(this).data('date');
                const rowPerson = $(this).data('person').toString().toLowerCase();
                const rowType = $(this).data('type');
                const rowMethod = $(this).data('method').toString().toLowerCase();
                const rowChequeStatus = $(this).data('cheque-status').toString().toLowerCase();
                const rowReferenceNumber = $(this).data('reference-number').toString().toLowerCase();
                const rowAmount = parseFloat($(this).data('amount-raw'));

                const datePass = (!startDate || rowDate >= startDate) && (!endDate || rowDate <= endDate);
                const personPass = !person || rowPerson.includes(person);
                const typePass = !type || rowType === type;
                const methodPass = !method || rowMethod === method;
                const chequeStatusPass = !chequeStatus || (rowChequeStatus && rowChequeStatus.includes(chequeStatus));
                const referenceNumberPass = !referenceNumber || rowReferenceNumber.includes(referenceNumber);

                if (datePass && personPass && typePass && methodPass && chequeStatusPass && referenceNumberPass) {{
                    $(this).show();
                    visibleRows++;
                    totalAmount += rowAmount;
                }} else {{
                    $(this).hide();
                }}
            }});
            $('#total-displayed-amount').text('Rs. ' + totalAmount.toLocaleString('en-IN', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}));
        }}

        function resetFilters() {{
            const today = new Date();
            const oneMonthAgo = new Date();
            oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
            $('#start-date').val(oneMonthAgo.toISOString().split('T')[0]);
            $('#end-date').val(today.toISOString().split('T')[0]);
            $('#name-filter').val('');
            $('#type-filter').val('');
            $('#method-filter').val('');
            $('#status-filter').val('');
            $('#reference-number-filter').val('');
            // Reset person options to All names when clearing type
            populatePersonOptions();
            applyFilters();
        }}
    </script>
</body>
</html>
"""
        if not os.path.exists(os.path.dirname(SUMMARY_FILE)):
            os.makedirs(os.path.dirname(SUMMARY_FILE))
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML summary generated at {SUMMARY_FILE}")
        st.toast("Summary updated successfully!")
    except Exception as e:
        st.error(f"Error generating summary: {e}")

# PDF Header for reports


def _generate_report_header(pdf: FPDF, title: str, person_name: str, start_date: datetime, end_date: datetime) -> None:
    """Generate header for PDF reports

    Args:
        pdf: FPDF instance
        title: Report title
        person_name: Name of person/client
        start_date: Start date of report period
        end_date: End date of report period
    """
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, title, 0, 1, 'C')
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 5, f"For: {person_name}", 0, 1, 'C')
    pdf.cell(
        0, 5, f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_draw_color(189, 195, 199)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

# PDF Footer for reports


def generate_pdf_footer(pdf):
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 10, f"Page {pdf.page_no()}/{{nb}}", border=0, align='C', new_x=XPos.RIGHT, new_y=YPos.TOP)

# Function to generate Inquiry Report (Payments Made to a person)
# Function to generate Bill Report (Client Expenses)
# Function to generate combined Invoice Report (Debits and Credits)
# Helper functions for CRUD operations


def save_transaction(new_row, payments_df):
    """Create: Add a new transaction"""
    try:
        new_payments_df = pd.concat(
            [payments_df, pd.DataFrame([new_row])], ignore_index=True)
        new_payments_df = DataCleaner.clean_payments_data(new_payments_df)
        write_csv_atomic(new_payments_df, CSV_FILE)
        generate_html_summary(new_payments_df)
        commit_and_push_files([CSV_FILE, SUMMARY_FILE], "Add transaction and update summary")
        st.success("✅ Transaction saved successfully!")
        return True
    except Exception as e:
        st.error(f"Error saving transaction: {str(e)}")
        return False


def update_transaction(index, updated_row, payments_df):
    """Update: Modify an existing transaction"""
    try:
        for col, value in updated_row.items():
            payments_df.loc[index, col] = value
        payments_df = DataCleaner.clean_payments_data(payments_df)
        write_csv_atomic(payments_df, CSV_FILE)
        generate_html_summary(payments_df)
        commit_and_push_files([CSV_FILE, SUMMARY_FILE], "Update transaction and update summary")
        st.success("✅ Transaction updated successfully!")
        return True
    except Exception as e:
        st.error(f"Error updating transaction: {str(e)}")
        return False


def delete_transaction(index, payments_df):
    """Delete: Remove an existing transaction"""
    try:
        payments_df = payments_df.drop(index).reset_index(drop=True)
        write_csv_atomic(payments_df, CSV_FILE)
        generate_html_summary(payments_df)
        commit_and_push_files([CSV_FILE, SUMMARY_FILE], "Delete transaction and update summary")
        st.success("✅ Transaction deleted successfully!")
        return True
    except Exception as e:
        st.error(f"Error deleting transaction: {str(e)}")
        return False


def save_client_expense(new_row, expenses_df):
    """Create: Add a new client expense"""
    try:
        # Robust date validation
        if not new_row.get('expense_date'):
            st.warning("Expense date is required.")
            return False
        new_expenses_df = pd.concat(
            [expenses_df, pd.DataFrame([new_row])], ignore_index=True)
        new_expenses_df = ExpenseCleaner.clean_client_expenses_data(
            new_expenses_df)
        write_csv_atomic(new_expenses_df, CLIENT_EXPENSES_FILE)
        commit_and_push_files([CLIENT_EXPENSES_FILE], "Add client expense")
        st.success("✅ Client expense saved successfully!")
        return True
    except Exception as e:
        st.error(f"Error saving client expense: {str(e)}")
        return False


def update_client_expense(index, updated_row, expenses_df):
    """Update: Modify an existing client expense"""
    try:
        for col, value in updated_row.items():
            expenses_df.loc[index, col] = value
        expenses_df = ExpenseCleaner.clean_client_expenses_data(expenses_df)
        write_csv_atomic(expenses_df, CLIENT_EXPENSES_FILE)
        commit_and_push_files([CLIENT_EXPENSES_FILE], "Update client expense")
        st.success("✅ Client expense updated successfully!")
        return True
    except Exception as e:
        st.error(f"Error updating client expense: {str(e)}")
        return False


def delete_client_expense(index, expenses_df):
    """Delete: Remove an existing client expense"""
    try:
        expenses_df = expenses_df.drop(index).reset_index(drop=True)
        write_csv_atomic(expenses_df, CLIENT_EXPENSES_FILE)
        commit_and_push_files([CLIENT_EXPENSES_FILE], "Delete client expense")
        st.success("✅ Client expense deleted successfully!")
        return True
    except Exception as e:
        st.error(f"Error deleting client expense: {str(e)}")
        return False


# Streamlit UI Configuration
st.set_page_config(
    layout="wide",
    page_title="Payments Tracker",
    page_icon="💸",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@example.com',
        'Report a bug': "mailto:bugs@example.com",
        'About': "# Payments Tracker\nA comprehensive solution for tracking payments and expenses."
    }
)

# Configure page layout and branding
st.markdown("""
    <style>
        .main > div {
            padding: 2rem 3rem;
        }
        .stTitle {
            font-size: 2.5rem !important;
            padding-bottom: 1rem;
            border-bottom: 2px solid #f0f2f6;
            margin-bottom: 2rem;
        }
        .stSidebar > div {
            padding: 2rem;
            background: #f8f9fa;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.title("💸 Payments Tracker")

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🎯 Quick Links")
    st.markdown(
        f"[🌐 View Public Summary]({SUMMARY_URL})", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    try:
        # Load recent statistics
        payments_df = pd.read_csv(CSV_FILE)
        total_transactions = len(payments_df)
        total_amount = payments_df['amount'].sum()
        st.metric("Total Transactions", f"{total_transactions:,}")
        st.metric("Total Amount", f"Rs. {total_amount:,.2f}")
    except Exception:
        st.warning("Statistics unavailable")

# Function to reset 'Add Transaction' form fields


def reset_form_session_state_for_add_transaction():
    st.session_state['selected_transaction_type'] = 'Paid to Me'
    # Also reset the radio's UI key to keep UI and logic in sync
    st.session_state['selected_transaction_type_radio'] = 'Paid to Me'
    st.session_state['payment_method'] = 'cash'
    st.session_state['selected_person'] = "Select..."
    st.session_state['editing_row_idx'] = None
    st.session_state['add_amount'] = None
    st.session_state['add_date'] = datetime.now().date()
    st.session_state['add_reference_number'] = ''
    st.session_state['add_cheque_status'] = 'received/given'
    st.session_state['add_status'] = 'completed'
    st.session_state['add_description'] = ''

# Function to reset 'Add Client Expense' form fields


def reset_form_session_state_for_add_client_expense():
    st.session_state['selected_client_for_expense'] = "Select..."
    st.session_state['add_client_expense_amount'] = None
    st.session_state['add_client_expense_date'] = datetime.now().date()
    st.session_state['add_client_expense_category'] = 'General'
    st.session_state['add_client_expense_description'] = ''
    st.session_state['add_client_expense_quantity'] = 1.0


# Apply form resets if triggered
if st.session_state.get('reset_add_form', False):
    reset_form_session_state_for_add_transaction()
    st.session_state['reset_add_form'] = False

if st.session_state.get('reset_client_expense_form', False):
    reset_form_session_state_for_add_client_expense()
    st.session_state['reset_client_expense_form'] = False

# Main tabs for navigation - Requested layout
transactions_tab, expenses_tab, reports_tab, manage_people_tab, backup_tab = st.tabs([
    "💰 Transactions",           # Add/Edit/Delete transactions
    "💳 Client Expenses",        # Add/Edit/Delete client expenses
    "📋 Reports & Analytics",    # Analysis and reporting
    "👥 Manage People",          # People management
    "🗄️ Backup & Restore"        # Backup and restore data
])

with transactions_tab:  # Core payment tracking functionality
    st.subheader("💰 Transaction Management")


    # Transaction Type Selection (outside form for real-time updates)
    st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <h4 style='margin:0 0 1rem 0; color: #2c3e50;'>📝 Transaction Type</h4>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Transaction type selection outside form for real-time updates
        selected_tx_type_local = st.radio(
            "💱 Transaction Type",
            ["Paid to Me", "I Paid"],
            horizontal=True,
            key='selected_transaction_type_radio',
            help="Select whether you received payment or made a payment"
        )
        st.session_state['selected_transaction_type'] = selected_tx_type_local
    with col2:
        payment_method_local = st.radio(
            "💳 Payment Method",
            ["cash", "cheque"],
            horizontal=True,
            key='payment_method_radio',
            help="Choose how the payment was made"
        )
        st.session_state['payment_method'] = payment_method_local

    # Load and filter people based on transaction type (outside form for real-time updates)
    try:
        people_df = pd.read_csv(PEOPLE_FILE)
        if 'category' in people_df.columns:
            # Normalize category and name fields
            cat_series = people_df['category'].astype(str).str.strip().str.lower()
            name_series = people_df['name'].astype(str).str.strip()

            # Use the radio button's current value directly
            selected_tx_type_effective = selected_tx_type_local.strip().lower()

            if selected_tx_type_effective == "i paid":
                mask = cat_series == 'client'
            elif selected_tx_type_effective == "paid to me":
                mask = cat_series == 'investor'
            else:
                mask = pd.Series([False] * len(people_df))

            filtered_people = name_series[mask].dropna().tolist()

            # Debug info to help troubleshoot
            unique_categories = cat_series.unique().tolist()
            st.info(f"Debug: Transaction type: '{selected_tx_type_effective}', Categories found: {unique_categories}, Filtered people count: {len(filtered_people)}")

            # If no matches found, show appropriate message and all people as fallback
            if len(filtered_people) == 0:
                if selected_tx_type_effective == "i paid":
                    st.warning("No clients found in people.csv. Showing all people as fallback.")
                elif selected_tx_type_effective == "paid to me":
                    st.warning("No investors found in people.csv. Showing all people as fallback.")
                filtered_people = name_series.dropna().tolist()
        else:
            # Fallback: if no category column, show all names
            filtered_people = people_df['name'].dropna().astype(str).str.strip().tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        st.error(f"Could not load people data file: {e}")
        filtered_people = []
    except pd.errors.ParserError as e:
        st.error(f"Error parsing people data: {e}")
        filtered_people = []

    person_options = ["Select..."] + sorted(filtered_people)

    if st.session_state['selected_person'] not in person_options:
        st.session_state['selected_person'] = "Select..."

    current_person_index = person_options.index(
        st.session_state['selected_person'])

    # Add Transaction form
    with st.form("transaction_form", clear_on_submit=False):
        st.subheader("➕ Add New Transaction")

        col1, col2 = st.columns(2)
        with col1:
            amount_value = st.session_state['add_amount']
            amount = st.number_input(
                "Amount (Rs.)",
                min_value=0.0,
                format="%.2f",
                value=(float(amount_value) if amount_value is not None else 0.0),
                key='add_amount'
            )

            date_value = st.session_state['add_date']
            date = st.date_input("Date", value=(date_value or datetime.now().date()), key='add_date')

            reference_number = st.text_input("Reference Number (Receipt/Cheque No.)",
                                             value=st.session_state['add_reference_number'], key='add_reference_number')

            cheque_status_val = ""
            if st.session_state['payment_method'] == "cheque":
                default_cheque_status_idx = 0
                if st.session_state['add_cheque_status'] in valid_cheque_statuses_lower:
                    default_cheque_status_idx = valid_cheque_statuses_lower.index(
                        st.session_state['add_cheque_status'])

                cheque_status_val = st.selectbox(
                    "Cheque Status",
                    valid_cheque_statuses_lower,
                    index=default_cheque_status_idx,
                    key='add_cheque_status'
                )

        with col2:
            st.session_state['selected_person'] = st.selectbox(
                "Select Person", person_options, index=current_person_index, key='selected_person_dropdown'
            )
            selected_person_final = st.session_state['selected_person'] if st.session_state[
                'selected_person'] != "Select..." else None

            default_status_idx = 0
            if st.session_state['add_status'] in valid_transaction_statuses_lower:
                default_status_idx = valid_transaction_statuses_lower.index(
                    st.session_state['add_status'])

            status = st.selectbox("Transaction Status", valid_transaction_statuses_lower,
                                  index=default_status_idx,
                                  key='add_status')
            description = st.text_input(
                "Description", value=st.session_state['add_description'], key='add_description')

        submitted = st.form_submit_button("Add Transaction")
        if submitted:
            validation_passed = True
            if not selected_person_final:
                st.warning("Please select a valid person.")
                validation_passed = False
            if amount is None or amount <= 0:
                st.warning("Amount must be greater than 0.")
                validation_passed = False

            normalized_reference_number = str(reference_number).strip()

            if not normalized_reference_number:
                st.warning(f"🚨 Reference Number is required.")
                validation_passed = False

            try:
                existing_df = pd.read_csv(
                    CSV_FILE, dtype={'reference_number': str}, keep_default_na=False)
                existing_df['reference_number'] = existing_df['reference_number'].apply(
                    lambda x: '' if pd.isna(x) or str(x).strip().lower() == 'nan' or str(
                        x).strip().lower() == 'none' else str(x).strip()
                )
                if not existing_df.empty and normalized_reference_number in existing_df['reference_number'].values:
                    st.warning(
                        f"🚨 Duplicate Reference Number found: '{normalized_reference_number}'. Please use a unique reference number.")
                    validation_passed = False
            except Exception as e:
                st.error(
                    f"Error checking for duplicate reference numbers: {e}")
                validation_passed = False

            if validation_passed:
                try:
                    new_row = {
                        "date": date.strftime("%Y-%m-%d"),
                        "person": selected_person_final,
                        "amount": amount,
                        "type": 'paid_to_me' if st.session_state['selected_transaction_type'] == "Paid to Me" else 'i_paid',
                        "status": status,
                        "description": description,
                        "payment_method": st.session_state['payment_method'],
                        "reference_number": normalized_reference_number,
                        "cheque_status": cheque_status_val if st.session_state['payment_method'] == "cheque" else None,
                        "transaction_status": status
                    }
                    # Load existing, append, clean, atomic write
                    try:
                        existing_df = pd.read_csv(
                            CSV_FILE, dtype={'reference_number': str}, keep_default_na=False)
                    except Exception:
                        existing_df = pd.DataFrame()

                    if not existing_df.empty and 'reference_number' in existing_df.columns:
                        existing_df['reference_number'] = existing_df['reference_number'].apply(
                            lambda x: '' if pd.isna(x) or str(x).strip().lower() == 'nan' or str(
                                x).strip().lower() == 'none' else str(x).strip()
                        )

                    # Avoid FutureWarning by aligning new row columns with existing_df before concat
                    new_row_df = pd.DataFrame([new_row]).reindex(columns=existing_df.columns)
                    updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                    updated_df = DataCleaner.clean_payments_data(updated_df)
                    write_csv_atomic(updated_df, CSV_FILE)

                    # Regenerate summary and auto-commit/push
                    try:
                        generate_html_summary(updated_df)
                        commit_and_push_files([CSV_FILE, SUMMARY_FILE], "Add transaction and update summary")
                        st.success("Transaction added successfully and GitHub updated.")
                    except Exception as e:
                        st.warning(f"Transaction added, but failed to update summary or push: {e}")

                    st.session_state['reset_add_form'] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving transaction: {e}")

    # Filters for viewing transactions (moved below Add Transaction)
    # Load people list for filter dropdowns and adapt person list to selected type
    try:
        people_df_all = pd.read_csv(PEOPLE_FILE)
    except Exception as e:
        st.warning(f"Could not load people for filter: {e}")
        people_df_all = pd.DataFrame(columns=["name", "category"])  # safe default

    col_type, col_person, col_method = st.columns(3)
    with col_type:
        view_type = st.selectbox("Filter by Type", ["All", "Paid to Me", "I Paid"], key="view_type_filter")

    # Build person options based on type selection
    if not people_df_all.empty and 'name' in people_df_all.columns:
        names_all = people_df_all['name'].astype(str).str.strip()
        if 'category' in people_df_all.columns:
            cat_series_all = people_df_all['category'].astype(str).str.strip().str.lower()
            if view_type == "I Paid":
                mask = cat_series_all == 'client'
                people_for_filter = names_all[mask].dropna().tolist()
            elif view_type == "Paid to Me":
                mask = cat_series_all == 'investor'
                people_for_filter = names_all[mask].dropna().tolist()
            else:
                # When view_type is All, show all people
                people_for_filter = names_all.dropna().tolist()
        else:
            # No category column; show all names
            people_for_filter = names_all.dropna().tolist()
    else:
        people_for_filter = []

    with col_person:
        view_person = st.selectbox("Filter by Person", ["All"] + sorted(people_for_filter), key="view_person_filter")
    with col_method:
        view_method = st.selectbox("Filter by Method", ["All", "cash", "cheque"], key="view_method_filter")

    # List, edit, and delete transactions (moved below add form)
    try:
        # Load and filter transactions
        transactions_df = pd.read_csv(
            CSV_FILE, dtype={'reference_number': str}, keep_default_na=False)
        filtered_df = transactions_df.copy()

        if view_person != "All":
            # Normalize spaces/case to ensure reliable matching
            filtered_df = filtered_df[
                filtered_df['person'].astype(str).str.strip() == str(view_person).strip()
            ]
        if view_type != "All":
            type_map = {"Paid to Me": "paid_to_me", "I Paid": "i_paid"}
            filtered_df = filtered_df[filtered_df['type'] == type_map.get(view_type, view_type)]
        if view_method != "All":
            filtered_df = filtered_df[filtered_df['payment_method'].str.lower() == view_method.lower()]

        # Display transactions with edit/delete buttons
        if not filtered_df.empty:
            for index, row in filtered_df.iterrows():
                with st.expander(f"Transaction: {row['person']} - Rs. {row['amount']} ({row['date']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Amount:** Rs. ", row['amount'])
                        st.write("**Date:** ", row['date'])
                        st.write("**Type:** ", "Received" if row['type'] == 'paid_to_me' else "Paid")
                        st.write("**Method:** ", row['payment_method'].capitalize())
                    with col2:
                        st.write("**Reference:** ", row['reference_number'])
                        st.write("**Status:** ", row['transaction_status'].capitalize())
                        if row['payment_method'].lower() == 'cheque':
                            st.write("**Cheque Status:** ", row['cheque_status'])
                        st.write("**Description:** ", row['description'])

                    edit_col, delete_col = st.columns(2)
                    with edit_col:
                        if st.button("✏️ Edit", key=f"edit_{index}"):
                            st.session_state['editing_row_idx'] = index
                            st.session_state['temp_edit_data'] = row.to_dict()
                            st.rerun()
                    with delete_col:
                        if st.button("🗑️ Delete", key=f"delete_{index}"):
                            if delete_transaction(index, transactions_df):
                                st.rerun()

            # Edit form (shows up when editing_row_idx is set)
            if st.session_state.get('editing_row_idx') is not None:
                with st.form(key='edit_transaction_form'):
                    st.subheader("Edit Transaction")
                    edit_data = st.session_state['temp_edit_data']

                    col1, col2 = st.columns(2)
                    with col1:
                        edited_amount = st.number_input("Amount", value=float(edit_data['amount']), min_value=0.0)
                        edited_date = st.date_input("Date", value=pd.to_datetime(edit_data['date']).date())
                        edited_payment_method = st.radio(
                            "Payment Method",
                            ["cash", "cheque"],
                            horizontal=True,
                            index=["cash", "cheque"].index(str(edit_data.get('payment_method', 'cash')).lower()),
                            key='edit_payment_method'
                        )

                        edited_cheque_status = ""
                        if edited_payment_method == "cheque":
                            cheque_status_val = str(edit_data.get('cheque_status', 'processing')).lower()
                            if cheque_status_val not in valid_cheque_statuses_lower:
                                cheque_status_val = 'processing'
                            edited_cheque_status = st.selectbox(
                                "Cheque Status",
                                valid_cheque_statuses_lower,
                                index=valid_cheque_statuses_lower.index(cheque_status_val),
                                key='edit_cheque_status'
                            )
                        else:
                            edited_cheque_status = None

                        edited_reference_number = st.text_input(
                            "Reference Number",
                            value=str(edit_data.get('reference_number', '')),
                            key='edit_reference_number'
                        )

                    with col2:
                        try:
                            people_df = pd.read_csv(PEOPLE_FILE)
                            people_list = people_df['name'].dropna().astype(str).tolist()
                            current_person = str(edit_data.get('person', ''))

                            if current_person not in people_list and current_person != '':
                                people_list = [current_person] + people_list
                                default_index = 0
                            elif current_person == '':
                                default_index = 0 if len(people_list) > 0 else 0
                            else:
                                default_index = people_list.index(current_person)

                            edited_person = st.selectbox(
                                "Select Person",
                                people_list,
                                index=default_index,
                                key='edit_person'
                            )
                        except Exception as e:
                            st.error(f"Error loading people data for edit: {e}")
                            edited_person = str(edit_data.get('person', ''))

                        transaction_type_options = {'Paid to Me': 'paid_to_me', 'I Paid': 'i_paid'}
                        edited_type_keys = list(transaction_type_options.keys())
                        current_type_val = next((key for key, val in transaction_type_options.items() if val == str(edit_data.get('type')).lower()), 'Paid to Me')
                        edited_type = st.radio("Transaction Type", edited_type_keys, horizontal=True, index=edited_type_keys.index(current_type_val), key='edit_type')

                        transaction_status_val = str(edit_data.get('transaction_status', 'completed')).lower()
                        if transaction_status_val not in valid_transaction_statuses_lower:
                            transaction_status_val = 'completed'
                        edited_transaction_status = st.selectbox(
                            "Transaction Status",
                            valid_transaction_statuses_lower,
                            index=valid_transaction_statuses_lower.index(transaction_status_val),
                            key='edit_transaction_status'
                        )

                        edited_description = st.text_input("Description", value=str(edit_data.get('description', '')), key='edit_description')

                    col1_btns, col2_btns, col3_btns = st.columns(3)
                    with col1_btns:
                        submit_button = st.form_submit_button("💾 Save Changes")
                    with col2_btns:
                        delete_button = st.form_submit_button("🗑️ Delete Transaction")
                    with col3_btns:
                        cancel_button = st.form_submit_button("❌ Cancel")

                    if submit_button:
                        validation_passed = True
                        if edited_amount is None or edited_amount <= 0:
                            st.warning("Amount must be greater than 0.")
                            validation_passed = False

                        normalized_edited_reference_number = str(edited_reference_number).strip()

                        if not normalized_edited_reference_number:
                            st.warning(f"🚨 Reference Number is required.")
                            validation_passed = False

                        try:
                            existing_df = pd.read_csv(CSV_FILE, dtype={'reference_number': str}, keep_default_na=False)
                            if 'reference_number' in existing_df.columns:
                                existing_df['reference_number'] = existing_df['reference_number'].apply(
                                    lambda x: '' if pd.isna(x) or str(x).strip().lower() == 'nan' or str(x).strip().lower() == 'none' else str(x).strip()
                                )
                            other_transactions = existing_df.drop(st.session_state['editing_row_idx'], errors='ignore')

                            if not other_transactions.empty and normalized_edited_reference_number in other_transactions['reference_number'].values:
                                st.warning(f"🚨 Duplicate Reference Number found: '{normalized_edited_reference_number}'. Please use a unique reference number.")
                                validation_passed = False
                        except Exception as e:
                            st.error(f"Error checking for duplicate reference numbers: {e}")
                            validation_passed = False

                        if validation_passed:
                            try:
                                updated_row = {
                                    "date": edited_date.strftime("%Y-%m-%d"),
                                    "person": edited_person,
                                    "amount": edited_amount,
                                    "type": transaction_type_options.get(edited_type, edited_type),
                                    "status": edited_transaction_status,
                                    "description": edited_description,
                                    "payment_method": edited_payment_method,
                                    "reference_number": normalized_edited_reference_number,
                                    "cheque_status": edited_cheque_status,
                                    "transaction_status": edited_transaction_status
                                }
                                if update_transaction(st.session_state['editing_row_idx'], updated_row, transactions_df):
                                    st.success("Transaction updated successfully!")
                                    st.session_state['editing_row_idx'] = None
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error updating transaction: {e}")
    except Exception as e:
        st.error(f"Error loading transactions: {e}")

with expenses_tab:  # Client Expenses
    st.subheader("💳 Client Expenses")

    try:
        people_df_for_expenses = pd.read_csv(PEOPLE_FILE)
        client_people = people_df_for_expenses[people_df_for_expenses['category'] == 'client']['name'].astype(
            str).tolist()
        client_options = ["Select..."] + sorted(client_people)

        current_client_index = 0
        if st.session_state['selected_client_for_expense'] in client_options:
            current_client_index = client_options.index(
                st.session_state['selected_client_for_expense'])

    except Exception as e:
        st.error(f"Error loading client data for expenses: {e}")
        client_options = ["Select..."]
        current_client_index = 0
        client_people = []

    # Category management (outside form)
    st.markdown("#### Manage Expense Categories")
    col_cat_mgmt1, col_cat_mgmt2 = st.columns([3, 1])
    with col_cat_mgmt1:
        if st.session_state.get('show_add_category_input', False):
            new_category = st.text_input("New Category Name", key='new_category_input', placeholder="e.g., Equipment")
    with col_cat_mgmt2:
        if not st.session_state.get('show_add_category_input', False):
            if st.button("➕ Add Category", key='show_add_cat_btn'):
                st.session_state['show_add_category_input'] = True
                st.rerun()
        else:
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("💾 Save", key='save_new_cat_btn'):
                    new_cat = st.session_state.get('new_category_input', '').strip()
                    if new_cat:
                        all_existing = valid_expense_categories + st.session_state.get('custom_expense_categories', [])
                        if new_cat not in all_existing:
                            if 'custom_expense_categories' not in st.session_state:
                                st.session_state['custom_expense_categories'] = []
                            st.session_state['custom_expense_categories'].append(new_cat)
                            st.session_state['add_client_expense_category'] = new_cat
                            st.session_state['show_add_category_input'] = False
                            st.success(f"Category '{new_cat}' added!")
                            st.rerun()
                        else:
                            st.warning("Category already exists!")
                    else:
                        st.warning("Please enter a category name.")
            with col_cancel:
                if st.button("❌ Cancel", key='cancel_new_cat_btn'):
                    st.session_state['show_add_category_input'] = False
                    st.rerun()

    st.markdown("---")
    st.markdown("#### Add New Expense")
    with st.form("client_expense_form", clear_on_submit=False):
        st.session_state['selected_client_for_expense'] = st.selectbox(
            "Select Client",
            client_options,
            index=current_client_index,
            key='selected_client_for_expense_select'
        )

        col1_exp, col2_exp = st.columns(2)
        with col1_exp:
            expense_amount_value = st.session_state['add_client_expense_amount']
            expense_amount = st.number_input(
                "Expense Amount (Unit Price Rs.)",
                min_value=0.0,
                format="%.2f",
                value=float(expense_amount_value) if expense_amount_value is not None else 0.0,
                key='add_client_expense_amount'
            )

            expense_date_value = st.session_state['add_client_expense_date']
            expense_date = st.date_input(
                "Expense Date", value=expense_date_value, key='add_client_expense_date')

            expense_quantity_value = st.session_state['add_client_expense_quantity']
            expense_quantity = st.number_input("Quantity", min_value=0.0, format="%.2f", value=float(
                expense_quantity_value) if expense_quantity_value is not None else 1.0, key='add_client_expense_quantity')

        with col2_exp:
            # Combine standard and custom categories only
            all_expense_categories = valid_expense_categories + st.session_state.get('custom_expense_categories', [])
            
            # Show current category selection
            current_category = st.session_state['add_client_expense_category']
            if current_category not in all_expense_categories:
                all_expense_categories.append(current_category)
            
            expense_category = st.selectbox(
                "Expense Category",
                all_expense_categories,
                index=all_expense_categories.index(current_category) if current_category in all_expense_categories else 0,
                key='add_client_expense_category'
            )
            
            expense_description = st.text_input(
                "Expense Description",
                value=st.session_state['add_client_expense_description'],
                key='add_client_expense_description'
            )
            original_transaction_ref_num = st.text_input(
                "Associated Transaction Ref. No. (Optional)", placeholder="e.g., invoice-123", key='add_client_expense_ref_num')

        submitted_expense = st.form_submit_button("Add Client Expense")

        if submitted_expense:
            expense_validation_passed = True
            selected_client_final_for_expense = None

            if st.session_state['selected_client_for_expense'] == "Select...":
                st.warning("Please select a client.")
                expense_validation_passed = False
            else:
                selected_client_final_for_expense = st.session_state['selected_client_for_expense']

            if expense_amount is None or expense_amount <= 0:
                st.warning(
                    "Expense amount (unit price) must be greater than 0.")
                expense_validation_passed = False
            if expense_quantity is None or expense_quantity <= 0:
                st.warning("Quantity must be greater than 0.")
                expense_validation_passed = False

            if expense_validation_passed:
                try:
                    new_expense_row = {
                        "original_transaction_ref_num": original_transaction_ref_num,
                        "expense_date": expense_date.strftime("%Y-%m-%d"),
                        "expense_person": selected_client_final_for_expense,
                        "expense_category": expense_category,
                        "expense_amount": expense_amount,
                        "expense_quantity": expense_quantity,
                        "expense_description": expense_description
                    }
                    # Load existing expenses, append, clean, atomic write
                    if not expense_date:
                        st.warning("Expense date is required.")
                        st.stop()

                    try:
                        existing_exp_df = pd.read_csv(
                            CLIENT_EXPENSES_FILE, keep_default_na=False)
                    except Exception:
                        existing_exp_df = pd.DataFrame()

                    updated_exp_df = pd.concat([existing_exp_df, pd.DataFrame([new_expense_row])], ignore_index=True)
                    updated_exp_df = ExpenseCleaner.clean_client_expenses_data(updated_exp_df)
                    write_csv_atomic(updated_exp_df, CLIENT_EXPENSES_FILE)

                    # For summary, load payments (do not change its data), then generate
                    try:
                        updated_payments_df = pd.read_csv(
                            CSV_FILE, dtype={'reference_number': str}, keep_default_na=False)
                        if 'reference_number' in updated_payments_df.columns:
                            updated_payments_df['reference_number'] = updated_payments_df['reference_number'].apply(
                                lambda x: '' if pd.isna(x) or str(x).strip().lower() == 'nan' or str(
                                    x).strip().lower() == 'none' else str(x).strip()
                            )
                        updated_payments_df = DataCleaner.clean_payments_data(updated_payments_df)
                    except Exception:
                        updated_payments_df = pd.DataFrame()

                    generate_html_summary(updated_payments_df)
                    commit_and_push_files([CLIENT_EXPENSES_FILE, SUMMARY_FILE], "Add client expense and update summary")

                    st.session_state['reset_client_expense_form'] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving client expense: {e}")

    # --- Client Expense History & Edit/Delete (moved from Manage People) ---
    st.markdown("---")
    st.subheader("📊 Client Expense History")

    # Add filters for client expenses
    st.markdown("**🔍 Filter Client Expenses**")
    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        expense_person = st.selectbox(
            "📋 Client",
            ["All"] + sorted(client_people),
            key="expense_person_filter",
            help="Filter expenses by specific client"
        )
    with col_exp2:
        # Include standard and custom categories only
        all_filter_categories = valid_expense_categories + st.session_state.get('custom_expense_categories', [])
        expense_category = st.selectbox(
            "📑 Category",
            ["All"] + all_filter_categories,
            key="expense_category_filter",
            help="Filter by expense category"
        )

    with col_exp3:
        expense_amount_range = st.select_slider(
            "💰 Amount Range (Rs.)",
            options=["0-10K", "10K-50K", "50K-100K",
                     "100K-500K", "500K-1M", "1M+", "All"],
            value="All",
            key="expense_amount_filter"
        )

    # Additional filters
    col_exp4, col_exp5 = st.columns(2)
    with col_exp4:
        quick_date_range = st.selectbox(
            "📅 Quick Date Range",
            ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "This Month", "This Year"],
            key="expense_quick_date_range"
        )
    with col_exp5:
        description_search = st.text_input("🔎 Description Contains", key="expense_description_search")

    col_exp6, col_exp7 = st.columns(2)
    with col_exp6:
        enable_custom_range = st.checkbox("Use Custom Date Range", value=False, key="expense_use_custom_range")
    if 'expense_start_date' not in st.session_state:
        st.session_state['expense_start_date'] = datetime.now().date().replace(day=1)
    if 'expense_end_date' not in st.session_state:
        st.session_state['expense_end_date'] = datetime.now().date()
    with col_exp6:
        start_date = st.date_input("Start Date", value=st.session_state['expense_start_date'], key="expense_start_date") if enable_custom_range else None
    with col_exp7:
        end_date = st.date_input("End Date", value=st.session_state['expense_end_date'], key="expense_end_date") if enable_custom_range else None

    reference_search = st.text_input("# Reference Contains", key="expense_reference_search")

    try:
        # Load and filter expenses
        expenses_df = pd.read_csv(CLIENT_EXPENSES_FILE,
                                  dtype={'original_transaction_ref_num': str,
                                         'expense_person': str},
                                  keep_default_na=False)
        expenses_df = clean_client_expenses_data(expenses_df)
        df_expenses = expenses_df.copy()
        filtered_df = expenses_df.copy()

        # Apply client filter
        if expense_person != "All":
            filtered_df = filtered_df[filtered_df['expense_person'] == expense_person]

        # Apply category filter
        if expense_category != "All":
            filtered_df = filtered_df[filtered_df['expense_category'] == expense_category]

        # Apply amount range filter
        if expense_amount_range != "All":
            amount_ranges = {
                "0-10K": (0, 10000),
                "10K-50K": (10000, 50000),
                "50K-100K": (50000, 100000),
                "100K-500K": (100000, 500000),
                "500K-1M": (500000, 1000000),
                "1M+": (1000000, float('inf'))
            }
            if expense_amount_range in amount_ranges:
                min_amount, max_amount = amount_ranges[expense_amount_range]
                filtered_df = filtered_df[
                    (filtered_df['expense_amount'] * filtered_df['expense_quantity'] >= min_amount) &
                    (filtered_df['expense_amount'] * filtered_df['expense_quantity'] < max_amount)
                ]

        # Apply date filters
        today = datetime.now().date()
        if quick_date_range != "All Time":
            if quick_date_range == "Last 7 Days":
                start_date = today - timedelta(days=7)
            elif quick_date_range == "Last 30 Days":
                start_date = today - timedelta(days=30)
            elif quick_date_range == "Last 90 Days":
                start_date = today - timedelta(days=90)
            elif quick_date_range == "This Month":
                start_date = today.replace(day=1)
            elif quick_date_range == "This Year":
                start_date = today.replace(month=1, day=1)

            filtered_df = filtered_df[
                (filtered_df['expense_date'] >= pd.Timestamp(start_date)) &
                (filtered_df['expense_date'] <= pd.Timestamp(today))
            ]

        # Apply custom date range if specified
        if start_date is not None:
            filtered_df = filtered_df[filtered_df['expense_date'] >= pd.Timestamp(start_date)]
        if end_date is not None:
            filtered_df = filtered_df[filtered_df['expense_date'] <= pd.Timestamp(end_date)]

        # Apply description search
        if description_search:
            filtered_df = filtered_df[
                filtered_df['expense_description'].str.lower().str.contains(description_search.lower(), na=False)
            ]

        # Apply reference number search
        if reference_search:
            filtered_df = filtered_df[
                filtered_df['original_transaction_ref_num'].str.lower().str.contains(reference_search.lower(), na=False)
            ]

        # Show active filters summary
        active_filters = []
        if expense_person != "All":
            active_filters.append(f"📋 Client: {expense_person}")
        if expense_category != "All":
            active_filters.append(f"📑 Category: {expense_category}")
        if expense_amount_range != "All":
            active_filters.append(f"💰 Amount: {expense_amount_range}")
        if quick_date_range != "All Time":
            active_filters.append(f"📅 Period: {quick_date_range}")
        if start_date or end_date:
            date_range = ""
            if start_date:
                date_range += f"From {start_date.strftime('%Y-%m-%d')}"
            if end_date:
                date_range += f" To {end_date.strftime('%Y-%m-%d')}"
            active_filters.append(f"📅 Custom Range: {date_range.strip()}")

        if active_filters:
            st.markdown("#### 🎯 Active Filters")
            for filter_info in active_filters:
                st.info(filter_info)

        # Show results summary
        total_records = len(filtered_df)
        if total_records > 0:
            total_amount = filtered_df['expense_amount'].multiply(filtered_df['expense_quantity']).sum()
            st.markdown(f"#### 📊 Results Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Expenses Found", f"{total_records:,}")
            with col2:
                st.metric("Total Amount", f"Rs. {total_amount:,.2f}")

        # Display expenses with edit/delete buttons
        if not filtered_df.empty:
            st.markdown("#### 📝 Expense Records")
            for index, row in filtered_df.iterrows():
                with st.expander(f"Expense: {row['expense_person']} - Rs. {row['expense_amount']} ({row['expense_date']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Amount:** Rs. ", row['expense_amount'])
                        st.write("**Quantity:** ", row['expense_quantity'])
                        st.write("**Total:** Rs. ", row['expense_amount'] * row['expense_quantity'])
                        st.write("**Date:** ", row['expense_date'])
                    with col2:
                        st.write("**Category:** ", row['expense_category'])
                        st.write("**Description:** ", row['expense_description'])
                        st.write("**Reference:** ", row['original_transaction_ref_num'])

                    edit_col, delete_col = st.columns(2)
                    with edit_col:
                        if st.button("✏️ Edit", key=f"edit_expense_{index}"):
                            st.session_state['editing_expense_row_idx'] = index
                            st.session_state['temp_edit_expense_data'] = row.to_dict()
                            st.rerun()
                    with delete_col:
                        if st.button("🗑️ Delete", key=f"delete_expense_{index}"):
                            if delete_client_expense(index, expenses_df):
                                st.rerun()

            # Edit form for expenses
            if st.session_state.get('editing_expense_row_idx') is not None:
                with st.form(key='edit_expense_form'):
                    st.subheader("Edit Client Expense")
                    edit_data = st.session_state['temp_edit_expense_data']

                    col1, col2 = st.columns(2)
                    with col1:
                        edited_expense_amount = st.number_input(
                            "Unit Amount (Rs.)",
                            value=float(edit_data['expense_amount']),
                            min_value=0.0,
                            format="%.2f",
                            key='edit_expense_amount'
                        )

                        default_expense_date_value = edit_data.get('expense_date')
                        if pd.isna(default_expense_date_value):
                            default_expense_date_value = None
                        elif isinstance(default_expense_date_value, pd.Timestamp):
                            default_expense_date_value = default_expense_date_value.date()
                        elif isinstance(default_expense_date_value, str):
                            try:
                                default_expense_date_value = datetime.strptime(default_expense_date_value, "%Y-%m-%d").date()
                            except ValueError:
                                default_expense_date_value = None

                        edited_expense_date = st.date_input(
                            "Date",
                            value=(default_expense_date_value or datetime.now().date()),
                            key='edit_expense_date'
                        )

                        edited_expense_quantity = st.number_input(
                            "Quantity",
                            value=float(edit_data.get('expense_quantity', 1.0)),
                            min_value=0.0,
                            format="%.2f",
                            key='edit_expense_quantity'
                        )

                    with col2:
                        try:
                            people_df_edit_exp = pd.read_csv(PEOPLE_FILE)
                            client_people_edit_exp = people_df_edit_exp[people_df_edit_exp['category'] == 'client']['name'].astype(str).tolist()
                            current_expense_person = str(edit_data.get('expense_person', ''))
                            if current_expense_person not in client_people_edit_exp and current_expense_person != '':
                                client_people_edit_exp = [current_expense_person] + client_people_edit_exp
                            client_people_edit_exp.sort()
                            edited_expense_person = st.selectbox(
                                "Client", client_people_edit_exp,
                                index=client_people_edit_exp.index(current_expense_person) if current_expense_person in client_people_edit_exp else 0,
                                key='edit_expense_person'
                            )
                        except Exception as e:
                            st.error(f"Error loading client data for expense edit: {e}")
                            edited_expense_person = str(edit_data.get('expense_person', ''))

                        # Include standard and custom categories only
                        all_edit_categories = valid_expense_categories + st.session_state.get('custom_expense_categories', [])
                        current_edit_category = str(edit_data.get('expense_category', 'General'))
                        if current_edit_category not in all_edit_categories:
                            all_edit_categories.append(current_edit_category)
                        edited_expense_category = st.selectbox(
                            "Category", all_edit_categories,
                            index=all_edit_categories.index(current_edit_category) if current_edit_category in all_edit_categories else 0,
                            key='edit_expense_category'
                        )
                        edited_expense_description = st.text_input(
                            "Description", value=str(edit_data.get('expense_description', '')),
                            key='edit_expense_description'
                        )
                        edited_original_transaction_ref_num = st.text_input(
                            "Associated Transaction Ref. No. (Optional)", value=str(edit_data.get('original_transaction_ref_num', '')),
                            key='edit_original_transaction_ref_num'
                        )

                    col1_exp_btns, col2_exp_btns, col3_exp_btns = st.columns(3)
                    with col1_exp_btns:
                        submit_expense_button = st.form_submit_button("💾 Save Changes")
                    with col2_exp_btns:
                        delete_expense_button = st.form_submit_button("🗑️ Delete Expense")
                    with col3_exp_btns:
                        cancel_expense_button = st.form_submit_button("❌ Cancel")

                    if submit_expense_button:
                        expense_validation_passed = True
                        if edited_expense_date is None:
                            st.warning("Expense date is required.")
                            expense_validation_passed = False
                        if edited_expense_amount is None or edited_expense_amount <= 0:
                            st.warning("Unit Amount must be greater than 0.")
                            expense_validation_passed = False
                        if edited_expense_quantity is None or edited_expense_quantity <= 0:
                            st.warning("Quantity must be greater than 0.")
                            expense_validation_passed = False

                        if expense_validation_passed:
                            try:
                                df_expenses.loc[st.session_state['editing_expense_row_idx']] = {
                                    "original_transaction_ref_num": edited_original_transaction_ref_num,
                                    "expense_date": edited_expense_date.strftime("%Y-%m-%d"),
                                    "expense_person": edited_expense_person,
                                    "expense_category": edited_expense_category,
                                    "expense_amount": edited_expense_amount,
                                    "expense_quantity": edited_expense_quantity,
                                    "expense_description": edited_expense_description
                                }
                                write_csv_atomic(df_expenses, CLIENT_EXPENSES_FILE)
                                commit_and_push_files([CLIENT_EXPENSES_FILE], "Update client expense")
                                st.success("✅ Client expense updated successfully!")
                                st.session_state['editing_expense_row_idx'] = None
                                st.session_state['temp_edit_expense_data'] = {}
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error saving client expense: {e}")

                    if delete_expense_button:
                        if st.session_state['editing_expense_row_idx'] is not None:
                            try:
                                st.warning(f"Are you sure you want to delete expense ID: {st.session_state['editing_expense_row_idx']}?")
                                if st.button("Confirm Deletion", key="confirm_delete_expense"):
                                    df_expenses.drop(st.session_state['editing_expense_row_idx'], inplace=True)
                                    write_csv_atomic(df_expenses, CLIENT_EXPENSES_FILE)
                                    commit_and_push_files([CLIENT_EXPENSES_FILE], "Delete client expense")
                                    st.success("🗑️ Client expense deleted successfully!")
                                    st.session_state['editing_expense_row_idx'] = None
                                    st.session_state['temp_edit_expense_data'] = {}
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting client expense: {e}")

                    if cancel_expense_button:
                        st.session_state['editing_expense_row_idx'] = None
                        st.session_state['temp_edit_expense_data'] = {}
                        st.rerun()

    except Exception as e:
        st.error(f"Error loading client expenses: {e}")


with manage_people_tab:  # Manage People
    st.subheader("👥 Manage People")

    # Load current people
    try:
        people_df = pd.read_csv(PEOPLE_FILE, keep_default_na=False)
        if 'name' not in people_df.columns or 'category' not in people_df.columns:
            people_df = pd.DataFrame(columns=['name', 'category'])
    except Exception:
        people_df = pd.DataFrame(columns=['name', 'category'])

    # Add form for new person
    with st.form("new_person_form", clear_on_submit=False):
        new_person_name = st.text_input("Name")
        new_person_category = st.selectbox("Category", ["client", "investor"], key='new_person_category')
        submitted_new_person = st.form_submit_button("Add Person")

    if submitted_new_person:
        name = (new_person_name or "").strip()
        if not name:
            st.warning("Name is required.")
        else:
            try:
                # Prevent exact duplicate (name + category)
                mask = (people_df['name'].astype(str).str.strip().str.lower() == name.lower()) & \
                       (people_df['category'].astype(str).str.strip().str.lower() == new_person_category.lower())
                if mask.any():
                    st.info("This person with the same category already exists.")
                else:
                    updated_people = pd.concat([
                        people_df,
                        pd.DataFrame([{"name": name, "category": new_person_category}])
                    ], ignore_index=True)
                    write_csv_atomic(updated_people, PEOPLE_FILE)
                    commit_and_push_files([PEOPLE_FILE], "Add person")
                    st.success("✅ Person added.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error adding person: {e}")

    st.markdown("---")
    st.markdown("#### Existing People")
    if people_df.empty:
        st.info("No people found. Add a client or investor above.")
    else:
        for idx, row in people_df.iterrows():
            with st.expander(f"{row['name']} ({row['category']})"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Name:** {row['name']}")
                    st.write(f"**Category:** {row['category']}")
                with col_b:
                    if st.button("🗑️ Delete", key=f"delete_person_{idx}"):
                        try:
                            updated_people = people_df.drop(idx).reset_index(drop=True)
                            write_csv_atomic(updated_people, PEOPLE_FILE)
                            commit_and_push_files([PEOPLE_FILE], "Delete person")
                            st.success("✅ Person deleted.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting person: {e}")

with reports_tab:  # Reports and analytics features
    st.subheader("📋 Reports & Analytics")
    st.info("Generate Inquiry, Bill, and Invoice reports with totals and net balance.")

    try:
        people_df_reports = pd.read_csv(PEOPLE_FILE)
        report_people = sorted(people_df_reports['name'].dropna().astype(str).tolist())
    except Exception:
        report_people = []

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        report_type = st.selectbox("Report Type", ["Inquiry", "Bill", "Invoice"], key="report_type_select")
    with col_r2:
        report_person = st.selectbox("Person/Client", ["Select..."] + report_people, key="report_person_select")
    with col_r3:
        # Show category filter only for Bill and Invoice reports
        # Include standard and custom categories only
        all_report_categories = valid_expense_categories + st.session_state.get('custom_expense_categories', [])
        if report_type in ["Bill", "Invoice"]:
            report_category = st.selectbox("Expense Category", ["All"] + all_report_categories, key="report_category_select")
        else:
            report_category = "All"
            st.selectbox("Expense Category", ["All"], key="report_category_select", disabled=True, help="Category filter only available for Bill and Invoice reports")
    with col_r4:
        today = datetime.now().date()
        default_start = today.replace(day=1)
        start_date = st.date_input("Start Date", value=default_start, key="report_start_date_input")
        end_date = st.date_input("End Date", value=today, key="report_end_date_input")

    col_rb1, col_rb2 = st.columns([1,2])
    with col_rb1:
        generate_btn = st.button("Generate Report (PDF)")

    if generate_btn:
        if report_person == "Select...":
            st.warning("Please select a person/client.")
        else:
            s = pd.Timestamp(start_date)
            e = pd.Timestamp(end_date)
            pdf_path = None
            # Get category filter value
            category_filter = report_category if report_type in ["Bill", "Invoice"] else None
            if report_type == "Inquiry":
                pdf_path = generate_inquiry_pdf(report_person, s, e)
            elif report_type == "Bill":
                pdf_path = generate_bill_pdf(report_person, s, e, category_filter)
            else:
                pdf_path = generate_invoice_pdf(report_person, s, e, category_filter)

            if pdf_path:
                st.success("Report generated successfully.")
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📥 Download PDF",
                        data=f,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf"
                    )

# Sidebar with summary statistics
st.sidebar.markdown("## 💰 Balance Summary")
try:
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        sidebar_df = pd.read_csv(
            CSV_FILE, dtype={'reference_number': str}, keep_default_na=False)
        sidebar_df['reference_number'] = sidebar_df['reference_number'].apply(
            lambda x: '' if pd.isna(x) or str(x).strip().lower() == 'nan' or str(
                x).strip().lower() == 'none' else str(x).strip()
        )
        sidebar_df = clean_payments_data(sidebar_df)

        if not sidebar_df.empty:
            # Calculate totals
            sidebar_df['amount'] = pd.to_numeric(
                sidebar_df['amount'], errors='coerce').fillna(0.0)
            sidebar_df['type'] = sidebar_df['type'].astype(str).str.lower()
            sidebar_df['payment_method'] = sidebar_df['payment_method'].astype(
                str).str.lower()

            total_received = sidebar_df[sidebar_df['type']
                                        == 'paid_to_me']['amount'].sum()
            total_paid = sidebar_df[sidebar_df['type']
                                    == 'i_paid']['amount'].sum()
            net_balance = total_paid - total_received

            # Calculate by payment method
            cash_received = sidebar_df[(sidebar_df['type'] == 'paid_to_me') & (
                sidebar_df['payment_method'] == 'cash')]['amount'].sum()
            cheque_received = sidebar_df[(sidebar_df['type'] == 'paid_to_me') & (
                sidebar_df['payment_method'] == 'cheque')]['amount'].sum()
            cash_paid = sidebar_df[(sidebar_df['type'] == 'i_paid') & (
                sidebar_df['payment_method'] == 'cash')]['amount'].sum()
            cheque_paid = sidebar_df[(sidebar_df['type'] == 'i_paid') & (
                sidebar_df['payment_method'] == 'cheque')]['amount'].sum()

            # Also compute total client expenses for sidebar metrics
            try:
                expenses_sidebar_df = pd.read_csv(CLIENT_EXPENSES_FILE, keep_default_na=False)
                expenses_sidebar_df = clean_client_expenses_data(expenses_sidebar_df)
                expenses_sidebar_df['line_total'] = pd.to_numeric(expenses_sidebar_df['expense_amount'], errors='coerce').fillna(0.0) * \
                    pd.to_numeric(expenses_sidebar_df['expense_quantity'], errors='coerce').fillna(1.0)
                total_client_expenses_for_sidebar = expenses_sidebar_df['line_total'].sum()
            except Exception:
                total_client_expenses_for_sidebar = 0.0

            # Display summary
            st.sidebar.metric("Total Received", f"Rs. {total_received:,.2f}")
            st.sidebar.metric("Total Paid (by me)", f"Rs. {total_paid:,.2f}")
            # Replace Net Balance metric with Total Client Expenses
            st.sidebar.metric("Total Client Expenses",
                              f"Rs. {total_client_expenses_for_sidebar:,.2f}")

            with st.sidebar.expander("Payment Methods"):
                st.write("**Received**")
                st.write(f"Cash: Rs. {cash_received:,.2f}")
                st.write(f"Cheque: Rs. {cheque_received:,.2f}")
                st.write("**Paid**")
                st.write(f"Cash: Rs. {cash_paid:,.2f}")
                st.write(f"Cheque: Rs. {cheque_paid:,.2f}")
        else:
            st.sidebar.info("No transactions yet.")
    else:
        st.sidebar.info(
            "Transaction database not found. Add a transaction to create it.")
except Exception as e:
    st.sidebar.error(f"Error loading balances: {str(e)}")

with backup_tab:  # System management features including backup/restore
    st.subheader("🔄 Backup & Restore Data")
    
    # Auto-update and push public HTML summary on each run (only writes when content changes)
    try:
        update_public_html_if_stale()
    except Exception:
        pass

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📦 Create Backup")
        st.info("Create a backup of all your data files (payments, expenses, people)")

        if st.button("Create Backup", type="primary"):
            backup_path = create_backup()
            if backup_path:
                st.success("✅ Backup created successfully!")
                with open(backup_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Backup File",
                        data=file,
                        file_name=os.path.basename(backup_path),
                        mime="application/zip"
                    )

    with col2:
        st.markdown("### 📤 Restore from Backup")
        st.info("Upload a backup file to restore your data")

        uploaded_file = st.file_uploader(
            "Choose backup file",
            type=["zip"],
            help="Select a backup ZIP file created by this application"
        )

        if uploaded_file is not None:
            if st.button("Restore Data", type="secondary"):
                restored_files = restore_backup(uploaded_file)
                if restored_files:
                    st.success(
                        f"✅ Successfully restored: {', '.join(restored_files)}")
                    st.info("🔄 Please refresh the page to see the restored data.")
                    st.balloons()
                else:
                    st.error(
                        "❌ Failed to restore backup. Please check the file format.")

    st.markdown("---")
    st.markdown("### ℹ️ Backup Information")
    st.markdown("""
    - **Backup includes**: All payment transactions, client expenses, and people data
    - **File format**: ZIP archive with CSV files
    - **Restore process**: Replaces current data with backup data
    - **Recommendation**: Create regular backups before making major changes
    """)

    # Removed Manage People section from Backup & Restore tab to keep responsibilities focused on backup/restore only.
