import sys
import re

with open(r'c:\New folder\Manga-translator_Gemini\mmt_gui\styles\dark.qss', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'QFrame#LeftToolBar\s*\{.*?(?=QPushButton:hover\s*\{)'
replacement = """QFrame#LeftToolBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #111116, stop:0.45 #0c0c10, stop:1 #060608);
}

QFrame#PageFilmstrip {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #111116, stop:0.5 #0a0a0c, stop:1 #060608);
    border-radius: 0px;
}

QFrame#StagePanel {
    background: #111116;
    border-radius: 0px;
}

QFrame#PreviewSurface {
    background: #060608;
    border-radius: 0px;
}

QFrame#SettingsCard,
QFrame#CollapsibleSection {
    background: #181820;
    border: 1px solid #2c2c36;
    border-radius: 0px;
}

QFrame#CollapsibleSectionBody {
    background: transparent;
    border: none;
}

QLabel[role="muted"] {
    color: #8a8a9e;
}

QLabel[role="title"] {
    font-family: "Orbitron", "Rajdhani", "Segoe UI", sans-serif;
    font-size: 16pt;
    font-weight: 700;
    color: #ffffff;
}

QLabel[role="sectionTitle"] {
    font-family: "Orbitron", "Rajdhani", "Segoe UI", sans-serif;
    font-size: 11pt;
    font-weight: 600;
    color: #e5e7eb;
}

QLabel[sectionBadge="true"] {
    background: #1f1133;
    color: #c084fc;
    border: 1px solid #6b21a8;
    border-radius: 0px;
    padding: 2px 8px;
    font-size: 9pt;
    font-weight: 600;
}

QToolButton[sectionToggle="true"] {
    background: transparent;
    border: none;
    color: #e5e7eb;
    font-size: 10.5pt;
    font-weight: 600;
    text-align: left;
    padding: 2px 0;
}

QToolButton[sectionToggle="true"]:hover {
    color: #ffd60a;
}

QPushButton,
QToolButton,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QLineEdit,
QPlainTextEdit,
QTextEdit {
    border: 1px solid #2c2c36;
    border-radius: 0px;
    background: #0a0a0c;
    color: #e5e7eb;
    padding: 6px 8px;
}

QLineEdit:disabled,
QComboBox:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QPlainTextEdit:disabled,
QTextEdit:disabled {
    background: #111116;
    color: #5a5a6e;
    border-color: #1a1a24;
}

QPushButton {
    min-height: 30px;
    border-top: 1px solid #4a4a5a;
    border-bottom: 1px solid #1a1a24;
    border-left: 1px solid #2c2c36;
    border-right: 1px solid #2c2c36;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #22222c, stop:1 #181820);
}

"""
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
with open(r'c:\New folder\Manga-translator_Gemini\mmt_gui\styles\dark.qss', 'w', encoding='utf-8') as f:
    f.write(new_content)
