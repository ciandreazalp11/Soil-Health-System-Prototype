                        "suitability_score",
                        "suitability_label",
                        "top_crops",
                        "verdict",
                    ]
                    if col in preview.columns
                ]
            ],
            use_container_width=True,
        )
        st.caption("**Soil health preview:** Each row shows the computed suitability score/label (Green/Orange/Red) and suggested crop groupings. Use this to verify the scoring makes sense before summarizing by province.")
        st.markdown("---")

        if "Province" in preview.columns:
            st.subheader("Per-province soil health summary")
            prov_summary = (
                preview.groupby("Province")
                .agg(
                    samples=("suitability_score", "count"),
                    avg_suitability=("suitability_score", "mean"),
                    green_samples=(
                        "suitability_label",
                        lambda x: (x == "Green").sum(),
                    ),
                    orange_samples=(
                        "suitability_label",
                        lambda x: (x == "Orange").sum(),
                    ),
                    red_samples=(
                        "suitability_label",
                        lambda x: (x == "Red").sum(),
                    ),
                )
                .reset_index()
            )
            prov_summary["avg_suitability"] = prov_summary["avg_suitability"].round(3)
            st.dataframe(prov_summary, use_container_width=True)
            st.caption("**Province summary:** Aggregates average suitability and counts per color. This helps compare areas and identify where soil improvement programs may be prioritized.")
            st.markdown("---")

        st.markdown("### Soil Suitability Color Legend")
        st.markdown(
            """
        <style>
        .legend-table {
            width: 97%;
            margin: 0 auto;
            background: rgba(255,255,255,0.06);
            border-radius: 11px;
            border: 1.4px solid #eee;
            box-shadow: 0 4px 16px #0001;
            font-size:17px;
        }
        .legend-table td {
            padding:10px 16px;
        }
        </style>
        <table class="legend-table">
          <tr>
            <td><span style="color:#2ecc71;font-weight:900;font-size:20px;">🟢 Green</span></td>
            <td><b>Good/Sustainable</b>. Soil is ideal for cropping.<br>
            <b>Recommended crops:</b> Rice, Corn, Cassava, Vegetables, Banana, Coconut.</td>
          </tr>
          <tr>
            <td><span style="color:#f39c12;font-weight:900;font-size:20px;">🟠 Orange</span></td>
            <td><b>Moderate</b>. Soil is OK but may require improvement.<br>
            <b>Actions:</b> Nutrient/fertilizer adjustment and checking pH.<br>
            <b>Crops:</b> Corn, Cassava, selected vegetables.</td>
          </tr>
          <tr>
            <td><span style="color:#e74c3c;font-weight:900;font-size:20px;">🔴 Red</span></td>
            <td><b>Poor/Unsuitable</b>. Not recommended for cropping.<br>
            <b>Actions:</b> Major improvement with organic matter, fertilizers, and pH correction.<br>
            <b>Crops:</b> Only hardy types after soil amendment.</td>
          </tr>
        </table>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---", unsafe_allow_html=True)

        st.subheader("Detailed crop evaluation for a specific soil sample")
        if df.shape[0] > 0:
            idx = st.number_input(
                "Select sample index (0-based)",
                min_value=0,
                max_value=int(df.shape[0] - 1),
                value=0,
                step=1,
            )
            sample_row = df.iloc[int(idx)]
            eval_df = build_crop_evaluation_table(sample_row, top_n=6)
            if not eval_df.empty:
                st.dataframe(eval_df, use_container_width=True)
                st.caption("**Crop suitability table:** Ranks crops for the selected sample based on your scoring logic. Higher scores indicate better match to the soil conditions represented by that row.")
            else:
                st.info(
                    "Unable to compute crop evaluation for this sample (missing values?)."
                )
        else:
            st.info("No samples available for detailed crop evaluation.")

        st.markdown("---")

        st.subheader("Soil pattern clustering (K-Means)")
        cluster_features = [f for f in features if f in df.columns]
        if len(cluster_features) < 2:
            st.info(
                "Need at least two numeric soil parameters (e.g., pH and Nitrogen) "
                "to run clustering."
            )
        else:
            n_clusters = st.slider("Number of clusters", 2, 5, 3, step=1)
            clustered_df, kmeans_model = run_kmeans_on_df(
                df, cluster_features, n_clusters=n_clusters
            )
            if clustered_df is None:
                st.info(
                    "Not enough valid rows to run clustering with the selected number of clusters."
                )
            else:
                counts = clustered_df["cluster"].value_counts().sort_index()
                st.write("Cluster sizes:")
                st.write(counts)

                x_feat = cluster_features[0]
                y_feat = cluster_features[1]
                fig_cluster = px.scatter(
                    clustered_df,
                    x=x_feat,
                    y=y_feat,
                    color="cluster",
                    title=f"K-Means clusters using {x_feat} vs {y_feat}",
                )
                fig_cluster.update_layout(template="plotly_dark")
                st.plotly_chart(fig_cluster, use_container_width=True)
                st.caption("**Clusters:** Each color is a group of samples with similar soil/environment patterns based on the selected features. Use this to discuss natural groupings or zones in your study area.")

elif page == "👤 About":
    st.title("👤 About the Makers")
    st.markdown("<div style='font-size:19px;'>Developed by:</div>", unsafe_allow_html=True)
    st.write("")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        render_profile("Andre Oneal A. Plaza", "andre_oneal_a._plaza.png")
    with col_b:
        render_profile("Rica Baliling", "rica_baliling.png")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:15px;color:#cd5fff;font-weight:600;'>All glory to God.</div>",
        unsafe_allow_html=True,
    )
