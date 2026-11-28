import json
import os
import re
import uuid

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")


def show_math(text):
    # Streamlit displays $...$ maths, but not the \( \) and \[ \] style some models use.
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n\n$$\n" + re.sub(r"\n\s*\n", "\n", m[1].strip()) + "\n$$\n\n",
                  text, flags=re.S)
    return re.sub(r"\\\((.+?)\\\)", lambda m: f"${m[1].strip()}$", text)


def add_pages(text, sources):
    """Replace passage numbers such as [2] with the file and page stored for that passage."""
    by_number = {s["n"]: s for s in sources}
    cited = []

    def label(match):
        numbers = [int(n) for n in match[1].split(",")]
        if any(n not in by_number for n in numbers):
            return match[0]  # not one of our passages: leave it unchanged
        cited.extend(n for n in numbers if n not in cited)
        places = dict.fromkeys(f"{by_number[n]['source']}, p. {by_number[n]['page']}" for n in numbers)
        return "[" + "; ".join(places) + "]"

    text = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", label, text)
    return text, [by_number[n] for n in sorted(cited)]


def show_answer(message):
    if message.get("query"):
        st.caption(f"Searched for: {message['query']}")
    if message.get("error"):
        if message["content"]:
            st.markdown(show_math(message["content"]))
        st.error(message["error"])
        return
    sources = message.get("sources", [])
    labelled, cited = add_pages(message["content"], sources)
    st.markdown(show_math(labelled))
    if sources:
        if cited:
            pages = dict.fromkeys(f"{s['source']}, page {s['page']}" for s in cited)
            st.caption("Cited: " + " | ".join(pages))
        else:
            st.warning("The answer has no valid citation. Check it against the document.")
        with st.expander("All retrieved passages (not all are cited)"):
            for s in sources:
                st.write(f"{s['source']}, page {s['page']} {s['section']}")


def stream_reply(question, source, earlier, reply, box):
    """Show the answer as it arrives and record it in `reply`."""
    try:
        body = {"question": question, "source": source, "history": earlier}
        with requests.post(f"{API}/ask", json=body,
                           stream=True, timeout=120) as r:
            r.raise_for_status()
            r.encoding = "utf-8"
            for line in r.iter_lines(decode_unicode=True):
                if not line.startswith("data: "):
                    continue
                e = json.loads(line[6:])
                if e["type"] == "sources":
                    reply["sources"] = e["data"]
                elif e["type"] == "query":
                    reply["query"] = e["data"]
                elif e["type"] == "error":
                    reply["error"] = e["data"]
                else:
                    reply["content"] += e["data"]
                    box.markdown(show_math(reply["content"]))
    except requests.RequestException as e:
        reply["error"] = f"Could not get an answer: {type(e).__name__}"


def save_chat():
    try:
        requests.put(f"{API}/history/{chat_id}", json=st.session_state.messages,
                     timeout=30).raise_for_status()
    except requests.RequestException:
        st.warning("This chat could not be saved, so a refresh may lose it.")


st.title("Document Q&A with citations")

# The chat id lives in the page address, so a refresh reopens the same chat.
if "chat" not in st.query_params:
    st.query_params["chat"] = uuid.uuid4().hex
chat_id = st.query_params["chat"]
if st.session_state.get("chat_id") != chat_id:
    try:
        r = requests.get(f"{API}/history/{chat_id}", timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        st.error("Could not load the chat. Wait a few seconds, then refresh.")
        st.stop()
    st.session_state.messages = r.json()
    st.session_state.chat_id = chat_id

with st.sidebar:
    file = st.file_uploader("Add a PDF", type="pdf")
    if file and st.button("Index it"):
        with st.spinner("Indexing... long PDFs can take several minutes."):
            try:
                r = requests.post(f"{API}/documents", files={"file": (file.name, file.getvalue())},
                                  timeout=600)
                st.success(f"{r.json()['chunks']} chunks indexed") if r.ok else st.error(r.text)
            except requests.Timeout:
                st.warning("Still indexing after 10 minutes. Refresh later to see if it appears.")
    docs = requests.get(f"{API}/documents", timeout=30).json()
    choice = st.selectbox("Search in", ["All documents", *docs])

    if docs:
        with st.expander("Delete a document"):
            target = st.selectbox("Document", docs, key="delete_target")
            sure = st.checkbox("Yes, delete it", key=f"sure_{target}")
            if st.button("Delete", disabled=not sure):
                r = requests.delete(f"{API}/documents", params={"source": target}, timeout=120)
                if r.ok:
                    del st.session_state[f"sure_{target}"]
                    st.rerun()
                st.error(r.text)

    if st.button("Clear chat", disabled=not st.session_state.messages):
        requests.delete(f"{API}/history/{chat_id}", timeout=30).raise_for_status()
        st.session_state.messages = []
        st.rerun()
    st.caption("Follow-ups use recent messages. Clear the chat when you change topic.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_answer(message)

question = st.chat_input("Ask about your documents")
if question:
    # recent successful messages only, taken before the new question is added
    earlier = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages if m["content"] and not m.get("error")][-6:]
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)
    save_chat()
    with st.chat_message("assistant"):
        box = st.empty()
        reply = {"role": "assistant", "content": "", "sources": []}
        stream_reply(question, None if choice == "All documents" else choice, earlier, reply, box)
        with box.container():
            show_answer(reply)
    st.session_state.messages.append(reply)
    save_chat()
    st.rerun()  # redraw so the sidebar buttons match the new chat
