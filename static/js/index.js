/**
 * Created by Administrator on 2017/5/17.
 */

$(function(){
    // index();
    $(".index_nav ul li").each(function(index){
        $(this).click(function(){
            console.log("xxxxxxx");
            var menuTitle = $(this).context.textContent;
            console.log(menuTitle);
            if(menuTitle == "热点关注"){
                window.location.href = "/rdgz_index"
            }else if(menuTitle == "情感分析"){
                window.location.href = "/qgfx_index"
            }else if(menuTitle == "主题检索"){
                window.location.href = "/ztjs_index"
            }else if(menuTitle == "互动体验"){
                window.location.href = "/hdty_index"
            }
        })
    });
});