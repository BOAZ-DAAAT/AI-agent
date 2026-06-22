from DATA_Analyst_Assistant_Agent.agents.sql.graph import build_app

app = build_app()

if __name__ == "__main__":
    while True:
        user_question = input("\n질문 입력 (종료하려면 exit): ").strip()
        if user_question.lower() == "exit":
            break

        result = app.invoke(
            {
                "user_question": user_question,
                "schema_text": "",
                "integrity_text": "",
                "plan": {},
                "sql_draft": {},
                "sql_result": None,
                "row_count": 0,
                "validation": {},
                "retry_count": 0,
                "max_retries": 2,
                "feedback": "",
                "error": "",
                "final_answer": "",
            }
        )

        print("\n" + "=" * 60)
        print("[Question]")
        print(user_question)

        print("\n[Plan]")
        print(result["plan"])

        print("\n[SQL]")
        print(result["sql_draft"]["sql"])

        print("\n[Validation]")
        print(result["validation"])

        print("\n[Final Answer]")
        print(result["final_answer"])
        print("=" * 60)
