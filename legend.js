/* The picture's words and the proper terms, one line each: the same legend on every page.
   LEGEND.rows(model) gives [[our word, the term, what it is], ...] with the model's own numbers filled in
   (model may be null: the numbers are then left out); LEGEND.mount(button) opens it as a dialog from a button. */
(function (root) {
  'use strict';
  const nf = n => n.toLocaleString('en-US');
  function rows(M) {
    const V = M ? nf(M.V) : 'tens of thousands', D = M ? M.D : 'a few hundred', K = M ? nf(M.weightBytes / 2) : 'millions', L = M ? M.L : 'several';
    return [
      ['piece', 'token', `one unit of text from the tokenizer's list; the model only ever sees its number (its ID). This model: ${V} of them, its vocabulary. Other models have their own lists, of other sizes.`],
      ['the dictionary, the list', 'vocabulary', 'every token a model can read or write.'],
      ['its spot in the cloud, its row of numbers', 'embedding (embedding vector)', `the ${D} numbers a token becomes; all rows together are the embedding matrix (${V} × ${D}). The cloud is a map of that matrix (t-SNE), so distances on it are rough.`],
      ['numbers per token', 'hidden size, d_model', `${D} in this model: the width of everything that flows through it.`],
      ['knob, dial', 'weight (a parameter)', `one number set in training and fixed after it; a grid of knobs is a weight matrix. ${K} in this model; "15M" and "70B" count these.`],
      ['layer, band', 'transformer layer (block)', `attention, then a feed-forward network, each with a rescaling step (RMSNorm or LayerNorm) before it; the result is added onto the token's numbers (the residual connection). ${L} of them here; big models have dozens.`],
      ['listening back', 'attention (self-attention)', "the newest token's query is scored against every earlier token's key; softmax turns the scores into attention weights (adding up to 100%); the earlier tokens' values are added up by those weights."],
      ['head', 'attention head', "one set of query, key, value and output weights doing attention in its own way; the heads' results are joined and added onto the token."],
      ['the stored earlier tokens', 'KV cache', "every earlier token's keys and values, kept in memory so they are not recomputed for each new token."],
      ['the knobs each token goes through alone', 'feed-forward network (MLP)', 'two or three weight matrices with a bend between them (SwiGLU or GELU), applied to each token separately.'],
      ["the ball, the card, the token's numbers", 'hidden state (residual stream)', `the ${D} numbers carried through the layers, changed a little by each one.`],
      ['the peek after each layer', 'logit lens', "running the final scoring on a layer's output to see what the model would say if it stopped there. A tool for looking; the model only scores at the end."],
      ['the wall of lights, the scores', 'logits', "one score per token: the final numbers (after the last rescaling) compared with each token's row of the output table (the unembedding, lm_head). Softmax turns them into probabilities: the lights' brightness."],
      ['the weighted dice', 'sampling (top-k, temperature)', 'keeping the most likely tokens (top-k) and drawing one by its probability; temperature rescales the scores first (0 = always the top one, higher = flatter).'],
      ['again, word by word', 'autoregressive generation', 'the chosen token joins the text and the whole model runs again for the next one; everything it reads so far is the context (window).'],
      ['the number becomes a word', 'detokenization', "the tokenizer's list read the other way round: the chosen token's number gives its text, and pieces are joined (a dot becomes a space)."],
      ['the hidden instructions', 'system prompt', 'text an app puts in front of your message; the model reads it as ordinary tokens.'],
      ['the program around the model', 'agent loop (tool use)', "a program people wrote that gives the model a task, reads its reply, runs any tool request it finds (a shell command, say), and feeds the result back as more tokens. The model only ever writes text."],
      ['set in training', 'training (gradient descent, backpropagation)', 'starting from random weights, nudging every weight a little toward the real next token, billions of times; the frozen result is the checkpoint you downloaded.'],
      ['in memory', 'RAM here; VRAM on a GPU in a data centre', 'the weights are loaded once and sit in memory while the model runs.'],
      ['the model', 'a language model (an LLM)', `a transformer trained to predict the next token${M ? `; this one is ${M.name}` : ''}. Big chat models are the same design at a far larger scale, with extra training on conversations.`],
    ];
  }
  const esc = s => String(s).replace(/[&<>]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
  let dlg = null;
  function open(getModel) {
    if (!dlg) {
      dlg = document.createElement('dialog'); dlg.className = 'lgdlg'; dlg.innerHTML = '<div class="lghead"><b>Our words, and the proper terms</b><button type="button" class="lgx" aria-label="Close">✕</button></div><div class="lgbody"></div>';
      const st = document.createElement('style'); st.textContent = `.lgdlg { border: 1px solid #262e45; border-radius: 16px; background: #0e111c; color: #d9dcf0; padding: 0; max-width: 640px; width: calc(100% - 32px); max-height: 80vh; font: 14.5px/1.5 'IBM Plex Sans', system-ui, sans-serif; } .lgdlg::backdrop { background: rgb(0 0 0 / .6); }
        .lghead { display: flex; justify-content: space-between; align-items: center; padding: 12px 14px 8px; } .lghead b { font-size: 15px; } .lgx { font: 600 15px system-ui; width: 32px; height: 32px; border-radius: 50%; border: 1.5px solid #262e45; background: #11141f; color: #eef0fa; cursor: pointer; }
        .lgbody { overflow-y: auto; max-height: calc(80vh - 56px); padding: 0 14px 14px; } .lgbody .pic { border-left: 3px solid #9d90ff; padding-left: 10px; color: #9aa1ba; font-size: 13.5px; margin: 0 0 10px; }
        .lgbody .lg { display: grid; gap: 8px; } .lgbody .lg div { border-bottom: 1px solid rgb(255 255 255 / .06); padding-bottom: 6px; } .lgbody .lg b { color: #ffe9b8; } .lgbody .lg i { font-style: normal; color: #8ff0ff; font-family: 'IBM Plex Mono', monospace; font-size: 13px; } .lgbody .lg small { display: block; color: #9aa1ba; }`;
      document.head.appendChild(st); document.body.appendChild(dlg);
      dlg.querySelector('.lgx').addEventListener('click', () => dlg.close()); dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });
    }
    const M = typeof getModel === 'function' ? getModel() : getModel;
    dlg.querySelector('.lgbody').innerHTML = `<p class="pic">Every page here uses one picture: knobs, a cloud, listening back, a wall of lights, a dice. Each stands for something exact in the model.</p><div class="lg">${rows(M).map(([a, b, d]) => `<div><b>${esc(a)}</b> → <i>${esc(b)}</i><small>${esc(d)}</small></div>`).join('')}</div>`;
    dlg.showModal();
  }
  function mount(button, getModel) { button.addEventListener('click', () => open(getModel)); }
  root.LEGEND = { rows, open, mount };
})(typeof window !== 'undefined' ? window : globalThis);
