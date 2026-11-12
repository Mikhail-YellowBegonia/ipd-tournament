"""锦标赛模拟器。接口与课程规则一致，未知细节见 TournamentConfig。"""

from ipd.sim.agent import Agent
from ipd.sim.config import TournamentConfig
from ipd.sim.tournament import TournamentResult, run_tournament

__all__ = ["Agent", "TournamentConfig", "TournamentResult", "run_tournament"]
