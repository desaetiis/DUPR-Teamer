import streamlit as st
import pandas as pd
import io
from pulp import pulp, LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, value
import contextlib
import sqlite3


def insert_players_into_db(players):
    conn = sqlite3.connect('players.db')
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute('DELETE FROM players')

    # Insert new data
    cursor.executemany('''
    INSERT INTO players (name, gender, dupr_rating)
    VALUES (?, ?, ?)
    ''', players)

    conn.commit()
    conn.close()


def generate_teams(players_df, pairing_option):
    """
    Generate teams of two players based on the given player data and pairing option.
    :param players_df: pandas DataFrame containing player data
    :param pairing_option: str, team pairing option
    :return: pandas DataFrame with the optimal teams
    """
    players = players_df['name'].tolist()
    ratings = dict(zip(players_df['name'], players_df['dupr_rating']))
    genders = dict(zip(players_df['name'], players_df['gender']))

    # Generate all possible pairs based on the pairing option
    pairs = []
    if pairing_option == "Any Gender":
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pairs.append((players[i], players[j]))
    elif pairing_option == "Mixed":
        males = [p for p in players if genders[p].lower() == 'male']
        females = [p for p in players if genders[p].lower() == 'female']
        for m in males:
            for f in females:
                pairs.append((m, f))
    elif pairing_option == "Gender":
        male_players = [p for p in players if genders[p].lower() == 'male']
        female_players = [p for p in players if genders[p].lower() == 'female']
        # Male-male pairs
        for i in range(len(male_players)):
            for j in range(i + 1, len(male_players)):
                pairs.append((male_players[i], male_players[j]))
        # Female-female pairs
        for i in range(len(female_players)):
            for j in range(i + 1, len(female_players)):
                pairs.append((female_players[i], female_players[j]))
    else:
        return None  # Invalid option

    # Check if it's possible to form teams under the selected option
    if len(pairs) == 0:
        st.error("No valid pairs can be formed with the selected pairing option.")
        return None

    # Decision variables: 1 if pair is selected, 0 otherwise
    pair_vars = LpVariable.dicts("pair", pairs, cat=LpBinary)

    # Initialize the problem
    prob = LpProblem("Team_Pairing", LpMinimize)

    # Objective: Minimize the difference between the highest and lowest team sums
    total_teams = len(players) // 2

    # Variables for max and min team sums
    max_team_sum = LpVariable("max_team_sum", lowBound=0)
    min_team_sum = LpVariable("min_team_sum", lowBound=0)

    # Constraints to set max and min team sums
    for pair in pairs:
        team_sum = ratings[pair[0]] + ratings[pair[1]]
        prob += max_team_sum >= team_sum * pair_vars[pair]
        prob += min_team_sum <= team_sum * pair_vars[pair] + 1000 * (1 - pair_vars[pair])  # Big M method

    #  The Big M method is a technique used to solve linear programming problems with constraints that are
    #  greater than or equal to zero. It's a modified version of the simplex algorithm that uses artificial
    #  variables and a large penalty value to find a solution

    # Objective function
    prob += max_team_sum - min_team_sum

    # Constraints: Each player is in exactly one team
    for player in players:
        prob += lpSum([pair_vars[pair] for pair in pairs if player in pair]) == 1

    # The number of teams should be total_players / 2
    prob += lpSum(pair_vars[pair] for pair in pairs) == total_teams

    # Solve the problem and capture solver output
    solver_output = io.StringIO()
    with contextlib.redirect_stdout(solver_output):
        result = prob.solve(pulp.PULP_CBC_CMD(msg=True))

    # Display the solver output in the Streamlit app (optional)
    # st.text("Solver Output:")
    # st.code(solver_output.getvalue())

    if result != 1:
        st.error("An error occurred while solving the optimization problem.")
        return None

    # Extract the teams
    team_list = []
    for pair in pairs:
        if value(pair_vars[pair]) == 1:
            team_rating_sum = ratings[pair[0]] + ratings[pair[1]]
            team_list.append({
                'Player 1': pair[0],
                'Player 2': pair[1],
                'Team Rating Sum': team_rating_sum,
                'Gender Composition': f"{genders[pair[0]]} & {genders[pair[1]]}"
            })

    teams_df = pd.DataFrame(team_list)
    teams_df = teams_df.sort_values(by='Team Rating Sum', ascending=False)
    return teams_df


def get_players_from_db(sample_data):
    conn = sqlite3.connect('players.db')
    if sample_data:
        tbl = "sample_players"
    else:
        tbl = "players"
    query = f"SELECT name, gender, dupr_rating FROM {tbl}"
    players_df = pd.read_sql_query(query, conn)
    conn.close()
    return players_df


def display_dataframe_in_expander(expander, df):
    with expander:
        st.dataframe(df)


def main():
    st.title("Pickleball Team Matcher")

    with st.expander("About", expanded=False):
        st.write("""
        Upload a CSV file containing player names, their DUPR ratings, and their gender,
        or enter the player data manually. The app will pair players into teams of two,
        minimizing the difference in team rating sums.

    Uses the PuLP library for linear programming optimization.

    `Author: Felix Vadan`
    """)

    data_option = st.selectbox(
        "How would you like to provide player data?",
        ("Enter Data Manually", "Upload CSV File", "Use Sample Data")
    )

    if data_option == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            players_df = pd.read_csv(uploaded_file)
            display_dataframe_in_expander(st.expander("Uploaded Data"), players_df)

    elif data_option == "Use Sample Data":
        players_df = get_players_from_db(sample_data=True)
        display_dataframe_in_expander(st.expander("Sample Data"), players_df)
    else:
        with st.expander(f"**Enter player details:**", expanded=True):
            # sample_data = [
            # st.write(f"**Enter player details:**")
            num_players = st.number_input("Number of players (must be even):", min_value=2, step=1, key="num_players",
                                          value=4)
            manual_data = []
            for i in range(int(num_players)):
                st.write(f"**Player {i + 1}**")
                name = st.text_input(f"Name of Player {i + 1}", key=f"name_{i}")
                rating = st.number_input(f"DUPR Rating of Player {i + 1}", min_value=0.0, step=0.01, key=f"rating_{i}")
                gender = st.selectbox(f"Gender of Player {i + 1}", options=["Male", "Female"], key=f"gender_{i}")
                manual_data.append((name, gender, rating))

        if st.button("Save Players"):
            insert_players_into_db(manual_data)
            st.success("Players saved to database.")
        players_df = pd.DataFrame(manual_data, columns=["name", "gender", "dupr_rating"])
        st.dataframe(players_df)

    if 'players_df' in locals():
        pairing_option = st.selectbox(
            "Select Team Pairing Option",
            ("Any Gender", "Mixed", "Gender")
        )

        if len(players_df) % 2 != 0:
            st.error("The number of players must be even.")
        else:
            if st.button("Generate Teams"):
                teams = generate_teams(players_df, pairing_option)

                if teams is not None:
                    st.write("**Optimal Teams:**")
                    st.dataframe(teams)
                else:
                    st.error("An error occurred while generating teams.")


if __name__ == "__main__":
    main()
