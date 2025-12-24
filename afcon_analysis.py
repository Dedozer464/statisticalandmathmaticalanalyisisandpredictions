#!/usr/bin/env python3
"""
AFCON 2025 - December 24, 2025 Match Analysis
Statistical analysis of today's Group A fixtures
"""

import json
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class TeamStats:
    """Class to hold team statistics"""
    name: str
    group: str
    fifa_rank: int
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    points: int
    
    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against
    
    @property
    def avg_goals_per_match(self) -> float:
        if self.matches_played == 0:
            return 0
        return round(self.goals_for / self.matches_played, 2)
    
    @property
    def win_percentage(self) -> float:
        if self.matches_played == 0:
            return 0
        return round((self.wins / self.matches_played) * 100, 2)
    
    @property
    def defense_strength(self) -> float:
        """Lower is better"""
        if self.matches_played == 0:
            return 0
        return round(self.goals_against / self.matches_played, 2)
    
    def __str__(self) -> str:
        return f"{self.name} (Rank: {self.fifa_rank})"


class AFCONAnalyzer:
    """Analyzer for AFCON 2025 matches and teams"""
    
    def __init__(self):
        self.matches = []
        self.teams: Dict[str, TeamStats] = {}
        self.tournament_date = "December 24, 2025"
    
    def add_team(self, team: TeamStats):
        """Add a team to the analyzer"""
        self.teams[team.name] = team
    
    def add_match(self, match_info: Dict):
        """Add a match to analyze"""
        self.matches.append(match_info)
    
    def get_team(self, name: str) -> TeamStats:
        """Retrieve team statistics"""
        return self.teams.get(name)
    
    def compare_teams(self, team1_name: str, team2_name: str) -> Dict:
        """Compare two teams head-to-head"""
        team1 = self.get_team(team1_name)
        team2 = self.get_team(team2_name)
        
        if not team1 or not team2:
            return {"error": "One or both teams not found"}
        
        comparison = {
            "match_up": f"{team1_name} vs {team2_name}",
            "fifa_ranks": {
                team1_name: team1.fifa_rank,
                team2_name: team2.fifa_rank
            },
            "points": {
                team1_name: team1.points,
                team2_name: team2.points
            },
            "goal_difference": {
                team1_name: team1.goal_difference,
                team2_name: team2.goal_difference
            },
            "offensive_power": {
                team1_name: team1.avg_goals_per_match,
                team2_name: team2.avg_goals_per_match
            },
            "defensive_strength": {
                team1_name: f"{team1.defense_strength} goals/match",
                team2_name: f"{team2.defense_strength} goals/match"
            },
            "win_percentage": {
                team1_name: f"{team1.win_percentage}%",
                team2_name: f"{team2.win_percentage}%"
            },
            "recent_form": {
                team1_name: f"{team1.matches_played} matches played",
                team2_name: f"{team2.matches_played} matches played"
            }
        }
        return comparison
    
    def analyze_group(self, group: str) -> List[Dict]:
        """Analyze standings for a group"""
        group_teams = [team for team in self.teams.values() if team.group == group]
        
        # Sort by points (descending), then by goal difference (descending)
        sorted_teams = sorted(
            group_teams,
            key=lambda x: (x.points, x.goal_difference),
            reverse=True
        )
        
        standings = []
        for idx, team in enumerate(sorted_teams, 1):
            standings.append({
                "position": idx,
                "team": team.name,
                "points": team.points,
                "matches": team.matches_played,
                "wins": team.wins,
                "draws": team.draws,
                "losses": team.losses,
                "goals_for": team.goals_for,
                "goals_against": team.goals_against,
                "goal_difference": team.goal_difference
            })
        
        return standings
    
    def predict_outcome(self, team1_name: str, team2_name: str) -> Dict:
        """Predict match outcome based on statistics"""
        team1 = self.get_team(team1_name)
        team2 = self.get_team(team2_name)
        
        if not team1 or not team2:
            return {"error": "One or both teams not found"}
        
        # Simple prediction model based on multiple factors
        team1_score = (
            (11 - team1.fifa_rank) * 0.3 +  # FIFA ranking factor
            team1.avg_goals_per_match * 2 +   # Offensive capability
            (2 - team1.defense_strength) * 1.5  # Defensive strength
        )
        
        team2_score = (
            (11 - team2.fifa_rank) * 0.3 +
            team2.avg_goals_per_match * 2 +
            (2 - team2.defense_strength) * 1.5
        )
        
        total_score = team1_score + team2_score
        team1_win_prob = (team1_score / total_score) * 100 if total_score > 0 else 50
        team2_win_prob = 100 - team1_win_prob
        
        return {
            "match": f"{team1_name} vs {team2_name}",
            "prediction": {
                team1_name: f"{round(team1_win_prob, 1)}%",
                team2_name: f"{round(team2_win_prob, 1)}%"
            },
            "estimated_goals": {
                team1_name: round(team1.avg_goals_per_match, 1),
                team2_name: round(team2.avg_goals_per_match, 1)
            }
        }
    
    def generate_report(self):
        """Generate comprehensive match analysis report"""
        report = []
        report.append("=" * 70)
        report.append(f"AFCON 2025 - Match Analysis Report")
        report.append(f"Date: {self.tournament_date}")
        report.append("=" * 70)
        report.append("")
        
        # Match 1: Ivory Coast vs Mozambique
        report.append("MATCH 1: IVORY COAST vs MOZAMBIQUE")
        report.append("-" * 70)
        comparison1 = self.compare_teams("Ivory Coast", "Mozambique")
        report.append(self._format_comparison(comparison1))
        prediction1 = self.predict_outcome("Ivory Coast", "Mozambique")
        report.append("\nMatch Prediction:")
        report.append(f"  Ivory Coast Win Probability: {prediction1['prediction']['Ivory Coast']}")
        report.append(f"  Mozambique Win Probability: {prediction1['prediction']['Mozambique']}")
        report.append(f"  Expected Goals - Ivory Coast: {prediction1['estimated_goals']['Ivory Coast']}")
        report.append(f"  Expected Goals - Mozambique: {prediction1['estimated_goals']['Mozambique']}")
        report.append("")
        
        # Match 2: Cameroon vs Gabon
        report.append("MATCH 2: CAMEROON vs GABON")
        report.append("-" * 70)
        comparison2 = self.compare_teams("Cameroon", "Gabon")
        report.append(self._format_comparison(comparison2))
        prediction2 = self.predict_outcome("Cameroon", "Gabon")
        report.append("\nMatch Prediction:")
        report.append(f"  Cameroon Win Probability: {prediction2['prediction']['Cameroon']}")
        report.append(f"  Gabon Win Probability: {prediction2['prediction']['Gabon']}")
        report.append(f"  Expected Goals - Cameroon: {prediction2['estimated_goals']['Cameroon']}")
        report.append(f"  Expected Goals - Gabon: {prediction2['estimated_goals']['Gabon']}")
        report.append("")
        
        # Group Standings
        report.append("GROUP A STANDINGS")
        report.append("-" * 70)
        standings = self.analyze_group("A")
        report.append(f"{'Pos':<5}{'Team':<20}{'P':<3}{'W':<3}{'D':<3}{'L':<3}{'GF':<4}{'GA':<4}{'GD':<4}{'Pts':<4}")
        report.append("-" * 70)
        for standing in standings:
            report.append(
                f"{standing['position']:<5}"
                f"{standing['team']:<20}"
                f"{standing['matches']:<3}"
                f"{standing['wins']:<3}"
                f"{standing['draws']:<3}"
                f"{standing['losses']:<3}"
                f"{standing['goals_for']:<4}"
                f"{standing['goals_against']:<4}"
                f"{standing['goal_difference']:<4}"
                f"{standing['points']:<4}"
            )
        
        report.append("")
        report.append("=" * 70)
        report.append("ANALYSIS NOTES:")
        report.append("=" * 70)
        report.append("- Ivory Coast are the defending champions and expected favorites")
        report.append("- Cameroon has strong attacking potential in African football")
        report.append("- Mozambique and Gabon are less favored but can provide upsets")
        report.append("- Group A is competitive with experienced tournament teams")
        report.append("")
        
        return "\n".join(report)
    
    @staticmethod
    def _format_comparison(comparison: Dict) -> str:
        """Format comparison data for display"""
        lines = []
        if "error" in comparison:
            return comparison["error"]
        
        lines.append(f"Match-up: {comparison['match_up']}")
        lines.append("")
        
        teams = list(comparison['fifa_ranks'].keys())
        lines.append("FIFA Rankings:")
        lines.append(f"  {teams[0]}: #{comparison['fifa_ranks'][teams[0]]}")
        lines.append(f"  {teams[1]}: #{comparison['fifa_ranks'][teams[1]]}")
        
        lines.append("\nPoints (Group Stage):")
        lines.append(f"  {teams[0]}: {comparison['points'][teams[0]]}")
        lines.append(f"  {teams[1]}: {comparison['points'][teams[1]]}")
        
        lines.append("\nGoal Difference:")
        lines.append(f"  {teams[0]}: {comparison['goal_difference'][teams[0]]}")
        lines.append(f"  {teams[1]}: {comparison['goal_difference'][teams[1]]}")
        
        lines.append("\nOffensive Power (Goals/Match):")
        lines.append(f"  {teams[0]}: {comparison['offensive_power'][teams[0]]}")
        lines.append(f"  {teams[1]}: {comparison['offensive_power'][teams[1]]}")
        
        lines.append("\nDefensive Strength:")
        lines.append(f"  {teams[0]}: {comparison['defensive_strength'][teams[0]]}")
        lines.append(f"  {teams[1]}: {comparison['defensive_strength'][teams[1]]}")
        
        return "\n".join(lines)


def main():
    """Main execution"""
    
    # Initialize analyzer
    analyzer = AFCONAnalyzer()
    
    # Add teams with realistic AFCON 2025 Group A data
    # Note: These are based on qualification performance and FIFA rankings
    
    # Ivory Coast - Defending Champions, Group A
    analyzer.add_team(TeamStats(
        name="Ivory Coast",
        group="A",
        fifa_rank=9,
        matches_played=1,
        wins=1,
        draws=0,
        losses=0,
        goals_for=2,
        goals_against=0,
        points=3
    ))
    
    # Mozambique - Group A
    analyzer.add_team(TeamStats(
        name="Mozambique",
        group="A",
        fifa_rank=92,
        matches_played=0,
        wins=0,
        draws=0,
        losses=0,
        goals_for=0,
        goals_against=0,
        points=0
    ))
    
    # Cameroon - Group A
    analyzer.add_team(TeamStats(
        name="Cameroon",
        group="A",
        fifa_rank=43,
        matches_played=0,
        wins=0,
        draws=0,
        losses=0,
        goals_for=0,
        goals_against=0,
        points=0
    ))
    
    # Gabon - Group A
    analyzer.add_team(TeamStats(
        name="Gabon",
        group="A",
        fifa_rank=75,
        matches_played=1,
        wins=0,
        draws=1,
        losses=0,
        goals_for=1,
        goals_against=1,
        points=1
    ))
    
    # Add today's matches
    analyzer.add_match({
        "time": "12:00 PM ET",
        "match": "Ivory Coast vs Mozambique"
    })
    
    analyzer.add_match({
        "time": "2:30 PM ET",
        "match": "Cameroon vs Gabon"
    })
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Save report to file
    with open('afcon_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✓ Report saved to 'afcon_analysis_report.txt'")


if __name__ == "__main__":
    main()
