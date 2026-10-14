"""Shared, accessible navigation dock styling for both football leagues."""

from urllib.parse import quote

_ICONS = {
    "week": '<rect x="3" y="5" width="18" height="16" rx="3"/><path d="M16 3v4M8 3v4M3 11h18M8 15h2M14 15h2"/>',
    "builder": '<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3"/><circle cx="15" cy="17" r="3"/>',
    "props": '<path d="M5 20V10M12 20V4M19 20v-7"/><circle cx="5" cy="7" r="1"/><circle cx="19" cy="10" r="1"/>',
    "rankings": '<path d="M8 21h8M12 17v4M7 4h10v6a5 5 0 0 1-10 0V4ZM7 6H4v3a4 4 0 0 0 4 4M17 6h3v3a4 4 0 0 1-4 4"/>',
    "results": '<rect x="4" y="3" width="16" height="18" rx="3"/><path d="m8 11 2 2 5-5M8 17h8"/>',
    "picks": '<path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2-5.6-3-5.6 3 1.1-6.2L3 9.6l6.2-.9L12 3Z"/>',
    "model": '<path d="m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5"/>',
}


def navigation_styles() -> str:
    icons = []
    for key, names in (
        ("active_screen", ("week", "builder", "props", "rankings", "results", "picks", "model")),
        ("cfb_active_screen", ("week", "builder", "rankings", "results", "picks")),
    ):
        for index, name in enumerate(names, 1):
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
                'fill="none" stroke="black" stroke-width="1.7" stroke-linecap="round" '
                f'stroke-linejoin="round">{_ICONS[name]}</svg>'
            )
            icons.append(
                f'.st-key-{key} [role="radiogroup"] label:nth-child({index}) '
                f'{{ --nav-icon: url("data:image/svg+xml,{quote(svg, safe="")}"); }}'
            )
    return "<style>" + _DOCK_CSS + "\n".join(icons) + "</style>"


_DOCK_CSS = """
    .st-key-active_screen, .st-key-cfb_active_screen {
      position: fixed;
      z-index: 1000;
      left: 50%;
      right: auto;
      bottom: max(14px, env(safe-area-inset-bottom));
      transform: translateX(-50%);
      width: min(760px, calc(100% - 32px)) !important;
      max-width: none !important;
      margin: 0 !important;
      padding: 7px !important;
      border: 1px solid rgba(226, 230, 235, .9);
      border-radius: 23px;
      background: rgba(255, 255, 255, .96);
      box-shadow: 0 8px 32px rgba(15, 23, 42, .10), 0 2px 6px rgba(15, 23, 42, .04);
      -webkit-backdrop-filter: blur(20px);
      backdrop-filter: blur(20px);
    }
    .st-key-cfb_active_screen { width: min(600px, calc(100% - 32px)) !important; }
    .st-key-active_screen > div, .st-key-cfb_active_screen > div { width: 100% !important; }
    .st-key-active_screen [role="radiogroup"], .st-key-cfb_active_screen [role="radiogroup"] {
      display: grid !important;
      width: 100% !important;
      gap: 3px !important;
    }
    .st-key-active_screen [role="radiogroup"] { grid-template-columns: repeat(7, minmax(0, 1fr)); }
    .st-key-cfb_active_screen [role="radiogroup"] { grid-template-columns: repeat(5, minmax(0, 1fr)); }
    .st-key-active_screen [role="radiogroup"] label,
    .st-key-cfb_active_screen [role="radiogroup"] label {
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      flex-direction: column !important;
      gap: 5px !important;
      min-width: 0 !important;
      min-height: 58px;
      width: 100% !important;
      margin: 0 !important;
      padding: 8px 1px !important;
      border: 0 !important;
      border-radius: 16px !important;
      background: transparent !important;
      color: #687384 !important;
      cursor: pointer;
      transition: background .18s ease, color .18s ease;
      -webkit-tap-highlight-color: transparent;
    }
    .st-key-active_screen [role="radiogroup"] label > div:first-child,
    .st-key-cfb_active_screen [role="radiogroup"] label > div:first-child { display: none !important; }
    .st-key-active_screen [role="radiogroup"] label::before,
    .st-key-cfb_active_screen [role="radiogroup"] label::before {
      content: '';
      display: block;
      width: 21px;
      height: 21px;
      flex: 0 0 21px;
      background: currentColor;
      -webkit-mask: var(--nav-icon) center / contain no-repeat;
      mask: var(--nav-icon) center / contain no-repeat;
    }
    .st-key-active_screen [role="radiogroup"] label p,
    .st-key-cfb_active_screen [role="radiogroup"] label p {
      margin: 0 !important;
      color: inherit !important;
      font-size: 11px !important;
      font-weight: 500;
      letter-spacing: -.15px;
      line-height: 1.2 !important;
      white-space: nowrap;
    }
    .st-key-active_screen [role="radiogroup"] label:has(input:checked),
    .st-key-cfb_active_screen [role="radiogroup"] label:has(input:checked) {
      color: #c74718 !important;
      background: #fff0e8 !important;
    }
    .st-key-active_screen [role="radiogroup"] label:has(input:checked) p,
    .st-key-cfb_active_screen [role="radiogroup"] label:has(input:checked) p { font-weight: 650; }
    .st-key-active_screen [role="radiogroup"] label:has(input:focus-visible),
    .st-key-cfb_active_screen [role="radiogroup"] label:has(input:focus-visible) {
      outline: 2px solid #c74718;
      outline-offset: 1px;
    }
    @media (hover: hover) {
      .st-key-active_screen [role="radiogroup"] label:hover,
      .st-key-cfb_active_screen [role="radiogroup"] label:hover { background: #f4f6f8 !important; }
      .st-key-active_screen [role="radiogroup"] label:has(input:checked):hover,
      .st-key-cfb_active_screen [role="radiogroup"] label:has(input:checked):hover { background: #ffe8da !important; }
    }
    @media (max-width: 520px) {
      .st-key-active_screen, .st-key-cfb_active_screen {
        width: calc(100% - 16px) !important;
        bottom: max(10px, env(safe-area-inset-bottom));
        padding: 5px !important;
        border-radius: 21px;
      }
      .st-key-active_screen [role="radiogroup"], .st-key-cfb_active_screen [role="radiogroup"] { gap: 1px !important; }
      .st-key-active_screen [role="radiogroup"] label p { font-size: 10px !important; }
    }
    @media (max-width: 360px) {
      .st-key-active_screen [role="radiogroup"] label p { font-size: 9px !important; }
    }
    @media (prefers-reduced-motion: reduce) {
      .st-key-active_screen [role="radiogroup"] label,
      .st-key-cfb_active_screen [role="radiogroup"] label { transition: none; }
    }
"""
