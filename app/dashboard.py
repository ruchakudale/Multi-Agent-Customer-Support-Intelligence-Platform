
import requests
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Customer Support Platform",
    page_icon="🎧",
    layout="wide",
)

API_BASE_URL = "http://127.0.0.1:8000"


# =========================================================
# HEADER
# =========================================================

st.title("AI Multi-Agent Customer Support Platform")

st.caption(
    "Intelligent ticket classification, priority prediction, "
    "FAQ retrieval, automated responses, and escalation monitoring."
)


# =========================================================
# API STATUS
# =========================================================

try:
    health_response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=5,
    )
    api_healthy = health_response.ok
except requests.RequestException:
    api_healthy = False

if api_healthy:
    st.success("FastAPI backend connected")
else:
    st.error(
        "FastAPI backend is unavailable. "
        "Start the API server before using the dashboard."
    )


# =========================================================
# SIDEBAR
# =========================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Submit Ticket",
        "Ticket History",
        "Human Review Queue",
        "Analytics"
    ]
)

# =========================================================
# SUBMIT TICKET
# =========================================================

if page == "Submit Ticket":

    st.subheader("Submit a customer support ticket")

    with st.form("ticket_form"):
        ticket_text = st.text_area(
            "Describe the customer's issue",
            placeholder=(
                "Example: My package has not arrived yet"
            ),
            height=140,
        )

        submitted = st.form_submit_button(
            "Process Ticket",
            type="primary",
        )

    if submitted:

        if not ticket_text.strip():
            st.warning("Please enter a ticket description.")

        elif not api_healthy:
            st.error("The FastAPI backend is not available.")

        else:
            with st.spinner(
                "Processing ticket through the AI agents..."
            ):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/tickets",
                        json={"ticket_text": ticket_text.strip()},
                        timeout=180,
                    )

                    if response.status_code == 201:
                        result = response.json()

                        st.session_state["last_ticket"] = result

                        st.success(
                            "Ticket processed and saved successfully."
                        )

                    else:
                        try:
                            detail = response.json().get(
                                "detail",
                                response.text,
                            )
                        except ValueError:
                            detail = response.text

                        st.error(
                            f"API error ({response.status_code}): "
                            f"{detail}"
                        )

                except requests.RequestException as exc:
                    st.error(f"Could not reach the API: {exc}")

    result = st.session_state.get("last_ticket")

    if result:
        st.divider()
        st.subheader("Ticket Results")

        st.code(result.get("ticket_id", "Unknown ticket ID"))

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Intent",
            result.get("intent") or "Unknown",
        )

        col2.metric(
            "Sentiment",
            result.get("sentiment") or "Unknown",
        )

        col3.metric(
            "Priority",
            result.get("priority") or "Unknown",
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Intent Confidence",
            (
                f"{result['intent_confidence']:.1%}"
                if result.get("intent_confidence") is not None
                else "N/A"
            ),
        )

        col2.metric(
            "Sentiment Confidence",
            (
                f"{result['sentiment_confidence']:.1%}"
                if result.get("sentiment_confidence") is not None
                else "N/A"
            ),
        )

        col3.metric(
            "Retrieval Score",
            (
                f"{result['retrieval_top_score']:.4f}"
                if result.get("retrieval_top_score") is not None
                else "N/A"
            ),
        )

        st.subheader("Suggested Customer Response")
        st.info(result.get("response") or "No response generated.")

        if result.get("escalated"):
            st.warning(
                "Escalation required: "
                + str(
                    result.get("escalation_reason")
                    or "Manual review recommended."
                )
            )
        else:
            st.success("No escalation triggered.")

        with st.expander("Agent Execution Logs"):
            st.json(result.get("agent_logs", []))


# =========================================================
# TICKET HISTORY
# =========================================================

elif page == "Ticket History":

    st.subheader("Recent Tickets")

    limit = st.slider(
        "Number of tickets",
        min_value=1,
        max_value=100,
        value=20,
    )

    if st.button("Load Ticket History"):

        try:
            response = requests.get(
                f"{API_BASE_URL}/tickets",
                params={"limit": limit},
                timeout=15,
            )
            response.raise_for_status()

            data = response.json()
            tickets = data.get("tickets", [])

            if tickets:
                st.dataframe(
                    tickets,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No tickets found.")

        except requests.RequestException as exc:
            st.error(f"Could not retrieve tickets: {exc}")



elif page == "Human Review Queue":
    st.subheader("Human Review Queue")
    st.caption(
        "Review AI-generated responses before approving or rejecting them."
    )

    reviewer = st.text_input(
        "Reviewer ID",
        value="support_agent_1",
        key="human_reviewer_id",
    ).strip()

    if st.button("Refresh Review Queue", type="primary"):
        st.session_state["refresh_review_queue"] = True

    try:
        response = requests.get(
            f"{API_BASE_URL}/human-review-queue",
            params={"limit": 100},
            timeout=15,
        )
        response.raise_for_status()

        data = response.json()
        tickets = data.get("tickets", [])

        st.metric("Pending Human Reviews", data.get("count", 0))

        if not tickets:
            st.success("No tickets are awaiting human review.")

        for ticket in tickets:
            ticket_id = ticket["ticket_id"]

            with st.expander(
                f'{ticket_id} | {ticket.get("priority", "Unknown")} Priority',
                expanded=True,
            ):
                st.write("**Customer issue**")
                st.write(ticket.get("ticket_text", ""))

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Predicted Intent",
                    ticket.get("intent") or "Unknown",
                )

                confidence = ticket.get("intent_confidence")
                col2.metric(
                    "Intent Confidence",
                    f"{confidence:.1%}" if confidence is not None else "N/A",
                )

                col3.metric(
                    "Priority",
                    ticket.get("priority") or "Unknown",
                )

                st.write("**Sentiment:**", ticket.get("sentiment", "Unknown"))
                st.write(
                    "**Escalation reason:**",
                    ticket.get("escalation_reason", "Not provided"),
                )

                st.write("**AI-suggested response**")
                st.info(ticket.get("response") or "No response available.")

                st.warning(
                    "Review the issue and suggested response before deciding. "
                    "Approval records your decision; it does not send the "
                    "response to the customer."
                )

                notes = st.text_area(
                    "Review notes",
                    key=f"review_notes_{ticket_id}",
                    placeholder="Explain your decision (optional for approval).",
                )

                col_approve, col_reject = st.columns(2)

                with col_approve:
                    approve_clicked = st.button(
                        "Approve Response",
                        key=f"approve_{ticket_id}",
                        type="primary",
                        use_container_width=True,
                    )

                with col_reject:
                    reject_clicked = st.button(
                        "Reject Response",
                        key=f"reject_{ticket_id}",
                        use_container_width=True,
                    )

                if approve_clicked or reject_clicked:
                    if not reviewer:
                        st.error("Enter a Reviewer ID before submitting a decision.")
                    elif reject_clicked and not notes.strip():
                        st.error("Please provide a reason when rejecting a response.")
                    else:
                        decision = (
                            "approved" if approve_clicked else "rejected"
                        )

                        payload = {
                            "decision": decision,
                            "reviewed_by": reviewer,
                            "review_notes": notes.strip(),
                        }

                        try:
                            review_response = requests.post(
                                f"{API_BASE_URL}/tickets/{ticket_id}/review",
                                json=payload,
                                timeout=15,
                            )

                            if review_response.ok:
                                result = review_response.json()

                                st.session_state[
                                    f"review_result_{ticket_id}"
                                ] = result

                                st.success(
                                    f"Ticket {ticket_id} marked "
                                    f"{result['status']}."
                                )
                                st.rerun()

                            else:
                                try:
                                    detail = review_response.json().get(
                                        "detail",
                                        review_response.text,
                                    )
                                except ValueError:
                                    detail = review_response.text

                                st.error(
                                    f"Review failed "
                                    f"({review_response.status_code}): {detail}"
                                )

                        except requests.RequestException as exc:
                            st.error(f"Could not submit review: {exc}")

    except requests.RequestException as exc:
        st.error(f"Could not load the human review queue: {exc}")


# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.subheader("Ticket Analytics")

    if st.button("Refresh Analytics", type="primary"):

        try:
            response = requests.get(
                f"{API_BASE_URL}/analytics",
                timeout=15,
            )
            response.raise_for_status()

            stats = response.json()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Tickets",
                stats.get("total_tickets", 0),
            )

            col2.metric(
                "Escalated Tickets",
                stats.get("escalated_tickets", 0),
            )

            col3.metric(
                "Escalation Rate",
                f"{stats.get('escalation_rate', 0):.1f}%",
            )


            st.subheader("Human-in-the-Loop Review Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Pending Reviews",
                stats.get("pending_human_reviews", 0),
            )

            col2.metric(
                "Approved Reviews",
                stats.get("approved_reviews", 0),
            )

            col3.metric(
                "Rejected Reviews",
                stats.get("rejected_reviews", 0),
            )

            priority_data = stats.get(
                "priority_distribution",
                {},
            )

            intent_data = stats.get(
                "intent_distribution",
                {},
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Priority Distribution")

                if priority_data:
                    st.bar_chart(priority_data)
                else:
                    st.info("No priority data available.")

            with col2:
                st.subheader("Intent Distribution")

                if intent_data:
                    st.bar_chart(intent_data)
                else:
                    st.info("No intent data available.")

        except requests.RequestException as exc:
            st.error(f"Could not retrieve analytics: {exc}")
