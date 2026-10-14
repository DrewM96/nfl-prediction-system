"""Presentation for the NFL player comparison board."""

from html import escape


def prop_card(row: dict, *, logo: str, age: str) -> str:
    direction = "over" if row["difference"] > 0 else "under" if row["difference"] < 0 else "even"
    freshness = f"{age} · older line" if row["stale"] else age
    books = int(row["books"])
    return (
        '<article class="prop-card">'
        '<div class="prop-player">'
        f'<div class="prop-logo">{logo}</div>'
        f'<div class="prop-identity"><h3>{escape(str(row["player_name"]))}</h3>'
        f'<div class="prop-matchup">{escape(str(row["position"]))} · '
        f"{escape(str(row['team']))} <span>vs</span> {escape(str(row['opponent']))}</div></div>"
        f'<span class="prop-lean prop-lean--{direction}">{escape(str(row["lean"]))}</span></div>'
        '<div class="prop-numbers">'
        f'<div><span class="prop-number-label">GRIDLINE</span><strong>{row["projection"]:.1f}</strong></div>'
        f'<div><span class="prop-number-label">Market line</span><strong>{row["line"]:.1f}</strong></div>'
        f'<div class="prop-difference prop-difference--{direction}"><span class="prop-number-label">Difference</span>'
        f"<strong>{row['difference']:+.1f}</strong></div></div>"
        '<div class="prop-card-footer">'
        f"<span>{books} {'book' if books == 1 else 'books'}</span>"
        f'<span class="{"prop-old" if row["stale"] else ""}">{escape(freshness)}</span></div>'
        "</article>"
    )


PROPS_STYLES = """
<style>
  .st-key-prop_type { width: 100% !important; max-width: 640px !important; }
  .st-key-prop_type > div { width: 100% !important; }
  .st-key-prop_type [role="radiogroup"] {
    display: grid !important;
    width: 100% !important;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 5px !important;
    padding: 5px;
    border: 1px solid #e9ecf0;
    border-radius: 18px;
    background: #f5f6f8;
    margin-bottom: 6px;
  }
  .st-key-prop_type [role="radiogroup"] label {
    width: 100% !important;
    min-width: 0;
    min-height: 44px;
    margin: 0 !important;
    padding: 10px 8px !important;
    border: 1px solid transparent !important;
    border-radius: 13px !important;
    justify-content: center;
    color: #657080 !important;
    background: transparent !important;
    transition: background .18s ease;
  }
  .st-key-prop_type [role="radiogroup"] label p { font-size: 13px !important; font-weight: 550; }
  .st-key-prop_type [role="radiogroup"] label:has(input:checked) {
    color: #bd4319 !important;
    background: white !important;
    border-color: #f2d9cc !important;
    box-shadow: 0 2px 5px rgba(15,23,42,.05);
  }
  .st-key-prop_type [role="radiogroup"] label:has(input:focus-visible) { outline: 2px solid #c74718; }
  .st-key-prop_board_controls {
    padding: 16px 18px;
    border: 1px solid #e9ecf0;
    border-radius: 18px;
    background: #fafbfc;
    margin-bottom: 2px;
  }
  .st-key-prop_board_controls [data-testid="stHorizontalBlock"] { align-items: center; }
  .st-key-prop_board_controls [data-testid="stSelectbox"] { max-width: 420px; }
  .st-key-prop_board_controls [data-testid="stCheckbox"] { padding-top: 20px; }
  .prop-board-heading {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    margin: 16px 0 14px;
  }
  .prop-board-heading h2 { margin: 0; padding: 0; font-size: 20px; font-weight: 600; letter-spacing: -.4px; }
  .prop-board-heading span { color: #77818f; font-size: 12px; white-space: nowrap; }
  .prop-board { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-bottom: 20px; }
  .prop-card {
    min-width: 0;
    padding: 20px;
    border: 1px solid #e7eaef;
    border-radius: 20px;
    background: white;
    box-shadow: 0 3px 12px rgba(15,23,42,.025);
  }
  .prop-player { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
  .prop-logo { display: grid; place-items: center; width: 44px; height: 44px; flex: 0 0 44px; border-radius: 13px; background: #f5f6f8; }
  .prop-logo img { width: 30px; height: 30px; object-fit: contain; }
  .prop-identity { min-width: 0; flex: 1; }
  .prop-identity h3 { margin: 0; padding: 0; font-size: 16px; font-weight: 650; line-height: 1.25; letter-spacing: -.25px; overflow-wrap: anywhere; }
  .prop-matchup { color: #687384; font-size: 12px; margin-top: 4px; }
  .prop-matchup span { color: #929aa6; margin: 0 2px; }
  .prop-lean { flex-shrink: 0; border-radius: 8px; padding: 5px 8px; font-size: 11px; font-weight: 650; }
  .prop-lean--over { color: #15765a; background: #eaf6ef; }
  .prop-lean--under { color: #5265a3; background: #eef1fb; }
  .prop-lean--even { color: #687384; background: #f1f3f5; }
  .prop-numbers { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); background: #f8f9fb; border: 1px solid #eef0f3; border-radius: 13px; padding: 13px 0; }
  .prop-numbers > div { min-width: 0; text-align: center; border-right: 1px solid #e8ebef; }
  .prop-numbers > div:last-child { border: 0; }
  .prop-number-label { display: block; font-size: 10px; letter-spacing: .25px; color: #6c7786; margin-bottom: 5px; }
  .prop-numbers strong { display: block; color: #18212c; font-size: 25px; font-weight: 600; letter-spacing: -.7px; line-height: 1.1; font-variant-numeric: tabular-nums; }
  .prop-difference--over strong { color: #15765a; }
  .prop-difference--under strong { color: #5265a3; }
  .prop-card-footer { display: flex; justify-content: space-between; gap: 8px; margin-top: 13px; font-size: 11px; color: #7a8492; }
  .prop-card-footer .prop-old { color: #a36315; }
  .prop-intro { color: #687384; font-size: 14px; margin: -5px 0 18px; line-height: 1.5; }
  @media (max-width: 760px) {
    .st-key-prop_type [role="radiogroup"] { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .st-key-prop_board_controls [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 10px; }
    .st-key-prop_board_controls [data-testid="stHorizontalBlock"] > div { width: 100% !important; flex: 1 1 100% !important; }
    .st-key-prop_board_controls [data-testid="stCheckbox"] { padding-top: 0; }
    .prop-board { grid-template-columns: 1fr; gap: 12px; }
    .prop-card { padding: 17px; }
  }
  @media (max-width: 360px) {
    .prop-player { gap: 9px; }
    .prop-identity h3 { font-size: 14px; }
    .prop-numbers strong { font-size: 22px; }
  }
  @media (prefers-reduced-motion: reduce) {
    .st-key-prop_type [role="radiogroup"] label { transition: none; }
  }
</style>
"""
