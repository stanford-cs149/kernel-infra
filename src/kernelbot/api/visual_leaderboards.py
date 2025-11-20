"""
Simple visual leaderboard HTML generator
"""

from typing import List
from libkernelbot.leaderboard_db import LeaderboardRankedEntry


def generate_simple_html(leaderboard_data: List[dict]) -> str:
    """
    Generate a simple HTML page displaying leaderboards.

    Args:
        leaderboard_data: List of dicts with keys: name, gpu_type, submissions

    Returns:
        HTML string
    """

    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Leaderboards</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }

        h1 {
            color: #333;
        }

        .leaderboard {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .leaderboard h2 {
            color: #555;
            margin-top: 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #333;
        }

        tr:hover {
            background-color: #f5f5f5;
        }

        .rank {
            font-weight: bold;
            color: #666;
        }

        .no-submissions {
            color: #999;
            font-style: italic;
            padding: 20px;
        }
    </style>
</head>
<body>
    <h1>GPU Leaderboards</h1>
"""

    # Generate a section for each leaderboard
    for lb in leaderboard_data:
        name = lb['name']
        gpu_type = lb['gpu_type']
        submissions = lb['submissions']

        html += f"""
    <div class="leaderboard">
        <h2>{name}</h2>
"""

        if not submissions:
            html += '        <p class="no-submissions">No submissions yet</p>\n'
        else:
            html += """
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>User</th>
                    <th>Submission</th>
                    <th>Score</th>
                    <th>Submitted</th>
                </tr>
            </thead>
            <tbody>
"""
            # Show submissions
            for sub in submissions:
                rank = sub['rank']
                user = sub['user_name']
                sub_name = sub['submission_name']
                score = sub['submission_score']
                time = sub['submission_time']

                html += f"""
                <tr>
                    <td class="rank">{rank}</td>
                    <td>{user}</td>
                    <td>{sub_name}</td>
                    <td>{score:.6f}s</td>
                    <td>{time}</td>
                </tr>
"""

            html += """
            </tbody>
        </table>
"""

        html += """
    </div>
"""

    html += """
</body>
</html>
"""

    return html
