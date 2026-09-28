"""Base classes for EdgeCraft tools."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from edgecraft.core.modality import Modality


class BaseTool(ABC):
    """Abstract base class for all EdgeCraft tools."""

    name: str = "base_tool"
    description: str = "Base tool"
    modality: Optional[Modality] = None  # None means global tool

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool.

        Returns:
            Dict with 'status', 'result', and optionally 'error'.
        """
        pass

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Allow calling tool as function."""
        return self.run(**kwargs)


class ToolRegistry:
    """Registry for managing tools across modalities."""

    _instance = None
    _tools: Dict[str, Dict[str, BaseTool]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tools = {"global": {}}
        return cls._instance

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register.
        """
        modality = tool.modality.value if tool.modality else "global"

        if modality not in cls._tools:
            cls._tools[modality] = {}

        cls._tools[modality][tool.name] = tool

    @classmethod
    def get(cls, name: str, modality: Modality = None) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Tool name.
            modality: Optional modality to search in first.

        Returns:
            Tool instance or None.
        """
        # Try modality-specific first
        if modality:
            mod_tools = cls._tools.get(modality.value, {})
            if name in mod_tools:
                return mod_tools[name]

        # Try global
        if name in cls._tools.get("global", {}):
            return cls._tools["global"][name]

        # Search all modalities
        for mod, tools in cls._tools.items():
            if name in tools:
                return tools[name]

        return None

    @classmethod
    def list_tools(cls, modality: Modality = None) -> List[str]:
        """List available tools.

        Args:
            modality: Optional modality filter.

        Returns:
            List of tool names.
        """
        if modality:
            return list(cls._tools.get(modality.value, {}).keys())

        all_tools = []
        for tools in cls._tools.values():
            all_tools.extend(tools.keys())
        return list(set(all_tools))

    @classmethod
    def get_tools_for_modality(cls, modality: Modality) -> Dict[str, BaseTool]:
        """Get all tools for a modality (including global).

        Args:
            modality: The modality.

        Returns:
            Dict of tool name to tool instance.
        """
        tools = {}
        tools.update(cls._tools.get("global", {}))
        tools.update(cls._tools.get(modality.value, {}))
        return tools
