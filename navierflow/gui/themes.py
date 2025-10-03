"""
Theme system for NavierFlow GUI
"""
from dataclasses import dataclass
from typing import Dict, Tuple
from enum import Enum
from PyQt6.QtGui import QPalette, QColor, QFont
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


class ThemeType(Enum):
    """Available theme types"""
    DARK = "dark"
    LIGHT = "light"
    BLUE = "blue"
    HIGH_CONTRAST = "high_contrast"
    CUSTOM = "custom"


@dataclass
class ColorScheme:
    """Color scheme definition"""
    # Primary colors
    primary: str
    primary_light: str
    primary_dark: str
    
    # Secondary colors
    secondary: str
    secondary_light: str
    secondary_dark: str
    
    # Background colors
    background: str
    background_light: str
    background_dark: str
    
    # Surface colors
    surface: str
    surface_light: str
    surface_dark: str
    
    # Text colors
    text_primary: str
    text_secondary: str
    text_disabled: str
    
    # Accent colors
    accent: str
    accent_light: str
    accent_dark: str
    
    # Status colors
    success: str
    warning: str
    error: str
    info: str
    
    # Border colors
    border: str
    border_light: str
    border_dark: str


class Theme:
    """Theme definition with colors and styles"""
    
    def __init__(self, name: str, theme_type: ThemeType, colors: ColorScheme):
        self.name = name
        self.type = theme_type
        self.colors = colors
    
    def apply_to_palette(self, palette: QPalette):
        """Apply theme colors to Qt palette"""
        # Window colors
        palette.setColor(QPalette.ColorRole.Window, QColor(self.colors.background))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(self.colors.text_primary))
        
        # Base colors
        palette.setColor(QPalette.ColorRole.Base, QColor(self.colors.surface))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(self.colors.surface_light))
        
        # Text colors
        palette.setColor(QPalette.ColorRole.Text, QColor(self.colors.text_primary))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(self.colors.text_secondary))
        
        # Button colors
        palette.setColor(QPalette.ColorRole.Button, QColor(self.colors.primary))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(self.colors.text_primary))
        
        # Highlight colors
        palette.setColor(QPalette.ColorRole.Highlight, QColor(self.colors.accent))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(self.colors.text_primary))
        
        # Disabled colors
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
                        QColor(self.colors.text_disabled))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
                        QColor(self.colors.text_disabled))
        
        return palette
    
    def get_stylesheet(self) -> str:
        """Get Qt stylesheet for the theme"""
        return f"""
        QMainWindow {{
            background-color: {self.colors.background};
            color: {self.colors.text_primary};
        }}
        
        QWidget {{
            background-color: {self.colors.background};
            color: {self.colors.text_primary};
        }}
        
        QPushButton {{
            background-color: {self.colors.primary};
            color: {self.colors.text_primary};
            border: 1px solid {self.colors.border};
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: {self.colors.primary_light};
        }}
        
        QPushButton:pressed {{
            background-color: {self.colors.primary_dark};
        }}
        
        QPushButton:disabled {{
            background-color: {self.colors.surface_dark};
            color: {self.colors.text_disabled};
        }}
        
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {self.colors.surface};
            color: {self.colors.text_primary};
            border: 1px solid {self.colors.border};
            border-radius: 4px;
            padding: 4px 8px;
        }}
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border: 2px solid {self.colors.accent};
        }}
        
        QComboBox {{
            background-color: {self.colors.surface};
            color: {self.colors.text_primary};
            border: 1px solid {self.colors.border};
            border-radius: 4px;
            padding: 4px 8px;
        }}
        
        QComboBox:hover {{
            border: 1px solid {self.colors.accent};
        }}
        
        QComboBox::drop-down {{
            border: none;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {self.colors.surface};
            color: {self.colors.text_primary};
            selection-background-color: {self.colors.accent};
        }}
        
        QSpinBox, QDoubleSpinBox {{
            background-color: {self.colors.surface};
            color: {self.colors.text_primary};
            border: 1px solid {self.colors.border};
            border-radius: 4px;
            padding: 4px 8px;
        }}
        
        QCheckBox, QRadioButton {{
            color: {self.colors.text_primary};
            spacing: 8px;
        }}
        
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {self.colors.border};
            border-radius: 3px;
            background-color: {self.colors.surface};
        }}
        
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background-color: {self.colors.accent};
        }}
        
        QSlider::groove:horizontal {{
            background: {self.colors.surface_dark};
            height: 8px;
            border-radius: 4px;
        }}
        
        QSlider::handle:horizontal {{
            background: {self.colors.accent};
            width: 18px;
            height: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background: {self.colors.accent_light};
        }}
        
        QProgressBar {{
            background-color: {self.colors.surface_dark};
            border: 1px solid {self.colors.border};
            border-radius: 4px;
            text-align: center;
            height: 20px;
        }}
        
        QProgressBar::chunk {{
            background-color: {self.colors.accent};
            border-radius: 3px;
        }}
        
        QTabWidget::pane {{
            border: 1px solid {self.colors.border};
            background-color: {self.colors.surface};
        }}
        
        QTabBar::tab {{
            background-color: {self.colors.surface_dark};
            color: {self.colors.text_secondary};
            border: 1px solid {self.colors.border};
            padding: 8px 16px;
            margin-right: 2px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {self.colors.accent};
            color: {self.colors.text_primary};
        }}
        
        QTabBar::tab:hover {{
            background-color: {self.colors.surface_light};
        }}
        
        QMenuBar {{
            background-color: {self.colors.background_dark};
            color: {self.colors.text_primary};
        }}
        
        QMenuBar::item:selected {{
            background-color: {self.colors.accent};
        }}
        
        QMenu {{
            background-color: {self.colors.surface};
            color: {self.colors.text_primary};
            border: 1px solid {self.colors.border};
        }}
        
        QMenu::item:selected {{
            background-color: {self.colors.accent};
        }}
        
        QToolBar {{
            background-color: {self.colors.background_dark};
            border: none;
            spacing: 4px;
            padding: 4px;
        }}
        
        QStatusBar {{
            background-color: {self.colors.background_dark};
            color: {self.colors.text_secondary};
        }}
        
        QScrollBar:vertical {{
            background: {self.colors.surface_dark};
            width: 12px;
            margin: 0px;
        }}
        
        QScrollBar::handle:vertical {{
            background: {self.colors.accent};
            min-height: 20px;
            border-radius: 6px;
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        
        QScrollBar:horizontal {{
            background: {self.colors.surface_dark};
            height: 12px;
            margin: 0px;
        }}
        
        QScrollBar::handle:horizontal {{
            background: {self.colors.accent};
            min-width: 20px;
            border-radius: 6px;
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        
        QGroupBox {{
            color: {self.colors.text_primary};
            border: 2px solid {self.colors.border};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 8px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            color: {self.colors.accent};
            font-weight: bold;
        }}
        
        QLabel {{
            color: {self.colors.text_primary};
        }}
        
        QToolTip {{
            background-color: {self.colors.surface_dark};
            color: {self.colors.text_primary};
            border: 1px solid {self.colors.border};
            border-radius: 4px;
            padding: 4px;
        }}
        """


class ThemeManager:
    """Manage application themes"""
    
    def __init__(self):
        self.themes: Dict[str, Theme] = {}
        self.current_theme: Theme = None
        self._initialize_themes()
    
    def _initialize_themes(self):
        """Initialize built-in themes"""
        
        # Dark Theme
        dark_colors = ColorScheme(
            primary="#2196F3",
            primary_light="#64B5F6",
            primary_dark="#1976D2",
            secondary="#9C27B0",
            secondary_light="#BA68C8",
            secondary_dark="#7B1FA2",
            background="#1E1E1E",
            background_light="#2D2D2D",
            background_dark="#121212",
            surface="#252525",
            surface_light="#303030",
            surface_dark="#1A1A1A",
            text_primary="#FFFFFF",
            text_secondary="#B0B0B0",
            text_disabled="#606060",
            accent="#4CAF50",
            accent_light="#81C784",
            accent_dark="#388E3C",
            success="#4CAF50",
            warning="#FFC107",
            error="#F44336",
            info="#2196F3",
            border="#404040",
            border_light="#505050",
            border_dark="#303030"
        )
        self.add_theme(Theme("Dark", ThemeType.DARK, dark_colors))
        
        # Light Theme
        light_colors = ColorScheme(
            primary="#2196F3",
            primary_light="#BBDEFB",
            primary_dark="#1976D2",
            secondary="#9C27B0",
            secondary_light="#E1BEE7",
            secondary_dark="#7B1FA2",
            background="#FAFAFA",
            background_light="#FFFFFF",
            background_dark="#F0F0F0",
            surface="#FFFFFF",
            surface_light="#F5F5F5",
            surface_dark="#E0E0E0",
            text_primary="#212121",
            text_secondary="#757575",
            text_disabled="#BDBDBD",
            accent="#4CAF50",
            accent_light="#C8E6C9",
            accent_dark="#388E3C",
            success="#4CAF50",
            warning="#FFC107",
            error="#F44336",
            info="#2196F3",
            border="#E0E0E0",
            border_light="#EEEEEE",
            border_dark="#BDBDBD"
        )
        self.add_theme(Theme("Light", ThemeType.LIGHT, light_colors))
        
        # Blue Theme
        blue_colors = ColorScheme(
            primary="#1E88E5",
            primary_light="#64B5F6",
            primary_dark="#1565C0",
            secondary="#00ACC1",
            secondary_light="#4DD0E1",
            secondary_dark="#00838F",
            background="#0D47A1",
            background_light="#1565C0",
            background_dark="#01579B",
            surface="#1976D2",
            surface_light="#2196F3",
            surface_dark="#1565C0",
            text_primary="#FFFFFF",
            text_secondary="#BBDEFB",
            text_disabled="#64B5F6",
            accent="#00BCD4",
            accent_light="#4DD0E1",
            accent_dark="#00838F",
            success="#4CAF50",
            warning="#FFC107",
            error="#F44336",
            info="#03A9F4",
            border="#1565C0",
            border_light="#1976D2",
            border_dark="#0D47A1"
        )
        self.add_theme(Theme("Blue", ThemeType.BLUE, blue_colors))
        
        # High Contrast Theme
        high_contrast_colors = ColorScheme(
            primary="#FFFF00",
            primary_light="#FFFF66",
            primary_dark="#CCCC00",
            secondary="#00FFFF",
            secondary_light="#66FFFF",
            secondary_dark="#00CCCC",
            background="#000000",
            background_light="#1A1A1A",
            background_dark="#000000",
            surface="#0A0A0A",
            surface_light="#1A1A1A",
            surface_dark="#000000",
            text_primary="#FFFFFF",
            text_secondary="#CCCCCC",
            text_disabled="#666666",
            accent="#00FF00",
            accent_light="#66FF66",
            accent_dark="#00CC00",
            success="#00FF00",
            warning="#FFFF00",
            error="#FF0000",
            info="#00FFFF",
            border="#FFFFFF",
            border_light="#CCCCCC",
            border_dark="#999999"
        )
        self.add_theme(Theme("High Contrast", ThemeType.HIGH_CONTRAST, high_contrast_colors))
        
        # Set default theme
        self.current_theme = self.themes["Dark"]
    
    def add_theme(self, theme: Theme):
        """Add a theme"""
        self.themes[theme.name] = theme
    
    def get_theme(self, name: str) -> Theme:
        """Get a theme by name"""
        return self.themes.get(name)
    
    def set_theme(self, name: str):
        """Set current theme"""
        if name in self.themes:
            self.current_theme = self.themes[name]
    
    def apply_theme(self, app: QApplication):
        """Apply current theme to application"""
        if self.current_theme:
            # Apply palette
            palette = QPalette()
            palette = self.current_theme.apply_to_palette(palette)
            app.setPalette(palette)
            
            # Apply stylesheet
            app.setStyleSheet(self.current_theme.get_stylesheet())
    
    def get_available_themes(self) -> list:
        """Get list of available theme names"""
        return list(self.themes.keys())
