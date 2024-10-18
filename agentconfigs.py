"""
Agent Configuration Module

This module provides abstract and concrete implementations of agent configurations
for use in simulations or experiments. It defines an abstract base class for agent
configurations and specific implementations for two agents: Willard and Jimmy.

The module also includes a utility function to retrieve the appropriate agent
configuration based on the agent's name.

Classes:
    AbstractAgentConfig: Abstract base class for agent configurations.
    WillardConfig: Configuration for the Willard agent.
    JimmyConfig: Configuration for the Jimmy agent.

Functions:
    get_agent_config: Retrieves the appropriate agent configuration.
"""
from abc import ABC, abstractmethod
from typing import Optional

class AbstractAgentConfig(ABC):
    """
    Abstract base class for agent configurations.

    This class defines the basic structure for agent configurations,
    including birth year and description attributes.
    """
    
    def __init__(self) -> None:
        self.birth_year: Optional[int] = None
        self.description: Optional[str] = None

class WillardConfig(AbstractAgentConfig):
    """
    Configuration class for the Willard agent.

    This class inherits from AbstractAgentConfig and provides specific
    details for the Willard character, including name, birth year, and description.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name: str = "Willard"
        self.birth_year: int = 1994 
        self.description: str = """
            Willard (Historian/Researcher in New Paltz, NY)
            Personality: Willard is thoughtful, reflective, and takes his work seriously. He's the type of person who values deep conversations but often struggles with expressing his emotions openly. While he loves his work researching obscure historical events, it often leaves him overworked and stressed. Willard is more reserved but enjoys Jimmy's company because Jimmy brings out a lighter side of him.
            Background: After graduating from the University of Maryland with a degree in History, Willard earned his master's in historical research. He now works at a small museum and spends a lot of his time digging through old archives, writing articles, and giving lectures. Willard recently bought a house in New Paltz with his partner (Hildur), and they are thinking about having kids. He's passionate about his career, but sometimes wonders if he's missing out on a more exciting life.
            Hobbies: Willard enjoys hiking, reading biographies, and occasionally writing for small history journals. He also has an old, beloved car that he likes to tinker with on weekends.
        """

class JimmyConfig(AbstractAgentConfig):
    """
    Configuration class for the Jimmy agent.

    This class inherits from AbstractAgentConfig and provides specific
    details for the Jimmy character, including name, birth year, and description.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name: str = "Jimmy"
        self.birth_year: int = 1994 
        self.description: str = """
            Jimmy (Trophy Shop Owner in Dryden, NY)
            Personality: Jimmy is a fun-loving, carefree guy who jokes around and doesn't take life too seriously. He's a bit of a goofball, and while his business is successful, he's always looking for ways to make things fun. He contrasts with Willard's serious demeanor but cares deeply about his friend.
            Background: Jimmy never thought he'd end up running a trophy shop, but it's what he inherited from his dad and turned into a thriving local business. He makes custom trophies, plaques, and awards for schools and businesses, and has a knack for customer service. Jimmy is married to Marge, and they have two young kids (Jimmy Jr, and Wendy). He loves his family but feels like he never has enough free time. Still, he wouldn't trade his life for anything.
            Hobbies: Jimmy likes bowling, hosting barbecues, and playing pranks on his friends. He's also really into fantasy football and is constantly trying to get Willard to join his league (even though Willard never does).        
        """

def get_agent_config(agent_name: str) -> AbstractAgentConfig:
    """
    Retrieves the appropriate agent configuration based on the given name.

    Args:
        agent_name (str): The name of the agent ("Willard" or "Jimmy").

    Returns:
        AbstractAgentConfig: An instance of the corresponding agent configuration.

    Raises:
        NotImplementedError: If an unsupported agent name is provided.
    """
    agent_name = agent_name.lower()
    
    if agent_name == "willard":
        return WillardConfig()
    elif agent_name == "jimmy":
        return JimmyConfig()
    else:
        raise NotImplementedError(f"Agent configuration for '{agent_name}' is not implemented.")




# 