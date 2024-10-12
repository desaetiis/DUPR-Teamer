import streamlit as st
import pandas as pd
import sys
from io import StringIO
from pulp import pulp, LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, value

# command to create the virtual environment
# python -m venv venv

# command to activate the virtual environment
# .\venv\Scripts\Activate

# command to create requirements.txt file
# pip freeze > requirements.txt

# command to install the requirements.txt file
# pip install -r requirements.txt

# command to run the app
# streamlit run app.py


def main():
    st.title("Pickleball Team Matcher")

    # add expander with about section
    with st.expander("About"):
        st.write("""
        Upload a CSV file containing player names, their DUPR ratings, and their gender,
        or enter the player data manually. The app will pair players into teams of two,
        minimizing the difference in team rating sums.
    
    Uses the PuLP library for linear programming optimization.
    
    Author: Felix Vadan
    """)

    # Option to upload CSV or enter data manually
    data_option = st.selectbox(
        "How would you like to provide player data?",
        ("Enter Data Manually", "Upload CSV File")
    )

    if data_option == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            players_df = pd.read_csv(uploaded_file)
            st.dataframe(players_df)
    else:
        # Enter data manually
        st.write("Enter player details:")
        num_players = st.number_input("Number of players (must be even):", min_value=2, step=1)
        manual_data = []
        for i in range(int(num_players)):
            st.write(f"**Player {i+1}**")
            name = st.text_input(f"Name of Player {i+1}", key=f"name_{i}")
            rating = st.number_input(f"DUPR Rating of Player {i+1}", min_value=0.0, step=0.01, key=f"rating_{i}")
            gender = st.selectbox(f"Gender of Player {i+1}", options=["Male", "Female"], key=f"gender_{i}")
            manual_data.append({"Name": name, "DUPR_Rating": rating, "Gender": gender})

        # Create DataFrame from manual input
        players_df = pd.DataFrame(manual_data)
        st.dataframe(players_df)

    if 'players_df' in locals():
        # Add selection for team pairing option
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
                    st.table(teams)
                else:
                    st.error("An error occurred while generating teams.")

def generate_teams(players_df, pairing_option):

    players = players_df['Name'].tolist()
    ratings = dict(zip(players_df['Name'], players_df['DUPR_Rating']))
    genders = dict(zip(players_df['Name'], players_df['Gender']))

    # Generate all possible pairs based on the pairing option
    pairs = []
    if pairing_option == "Any Gender":
        for i in range(len(players)):
            for j in range(i+1, len(players)):
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
            for j in range(i+1, len(male_players)):
                pairs.append((male_players[i], male_players[j]))
        # Female-female pairs
        for i in range(len(female_players)):
            for j in range(i+1, len(female_players)):
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

    # Objective function
    prob += max_team_sum - min_team_sum

    # Constraints: Each player is in exactly one team
    for player in players:
        prob += lpSum([pair_vars[pair] for pair in pairs if player in pair]) == 1

    # The number of teams should be total_players / 2
    prob += lpSum(pair_vars[pair] for pair in pairs) == total_teams

    # Solve the problem and capture solver output
    import contextlib
    import io

    solver_output = io.StringIO()
    with contextlib.redirect_stdout(solver_output):
        result = prob.solve(pulp.PULP_CBC_CMD(msg=True))

    # Display the solver output in the Streamlit app
    st.text("Solver Output:")
    st.code(solver_output.getvalue())

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

if __name__ == "__main__":
    main()
